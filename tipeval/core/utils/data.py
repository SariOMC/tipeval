"""This module contains some utility functions used for things such as data reading or data conversion. """
from __future__ import annotations

from datetime import date
from functools import partial
from importlib.resources import path, Path, Package
import os
import pickle
import typing as T

import h5py
import numpy as np
from skimage import measure

from tipeval.core.typing import FilePath, Point2D

if T.TYPE_CHECKING:
    from tipeval.core.tips import Tip


TO_UNIT = dict(m=1.0, mm=1e3, µm=1e6, nm=1e9, Å=1e10)
FROM_UNIT = {k: 1.0 / v for k, v in TO_UNIT.items()}

BASE_UNIT = 'nm'


def reshape(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> T.Iterator[np.ndarray]:
    """
    Reshape flattened 1D xyz arrays into the image dimensions used during data recording.

    The shape is defined by the unique values in x and y. If the precision of the input
    data is too low it may happen that some x and y values occur more often than in the
    actual image. Therefore, the number of points in x and y are underestimated.

    :param x: the x data
    :param y: the y data
    :param z: the z data
    :return: the three original 1D arrays reshaped into 2D arrays
    """

    shape = len(np.unique(y)), len(np.unique(x))
    try:
        reshaped = tuple(map(lambda arr: np.reshape(arr, shape), (x, y, z)))
    except ValueError:
        raise ValueError('Cannot reshape array. Possibly the precision of input array is too low so that '
                         'the number unique values of x and y are underestimated. ')
    return reshaped


def conversion_factor(fu='m', tu='m'):
    """
    Function used to convert between units:

    The available units can be found in the TO_UNIT and FROM_UNIT dictionaries.

    :param fu: the unit from which to convert
    :param tu: the unit to which to convert
    :return: the resulting conversion factor
    """

    return FROM_UNIT[fu] * TO_UNIT[tu]


"""Convenience function for converting to the base unit. """
to_base = partial(conversion_factor, tu='nm')

"""Convenience function for converting from the base unit. """
from_base = partial(conversion_factor, fu='nm')


def load_data(file_information):
    delimiter = file_information.pop('delimiter')

    if delimiter == 'tab':
        delimiter = '\t'
    if delimiter == 'space':
        delimiter = ' '

    comment_symbol = file_information.pop('comment_symbol')
    file = file_information.get('input_file')
    from_unit = file_information.pop('base_unit')
    to_unit = file_information.get('new_unit')

    data = np.loadtxt(file, delimiter=delimiter, comments=comment_symbol,
                      unpack=True)
    data *= conversion_factor(from_unit, to_unit)

    x, y, z = reshape(*data)

    new_data = move_data_center(x, y, z)

    return new_data


def move_data_center(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Shift the x and y data such that the maximum z is at x, y = 0, 0

    :param x, y, z: the input coordinates
    :return: a numpy array containing the converted coordinates
    """

    max_z, (index_max_x, index_max_y) = find_max_image(z, range=(0.25, 0.75))

    maximum_x, maximum_y = x[index_max_x, index_max_y], y[index_max_x, index_max_y]

    z_conv = (z - max_z) * -1
    x_conv = x - maximum_x
    y_conv = y - maximum_y

    return np.array([x_conv, y_conv, z_conv])


def parse_time(date: date) -> str:
    from tipeval.config import configuration
    fmt = configuration.date_format
    return date.strftime(fmt)


def find_max_image(arr: np.ndarray, range: T.Tuple[float, ...] = (0., 1.)) -> T.Tuple[float, T.Tuple[np.ndarray]]:
    """Get the maximum and indices of maximum in a given array.

    This function searches for the maximum value in a given array. By using the 'range' variable one
    can choose a certain range of the array wherein the maximum will be searched. 'range' can be a
    tuple containing two floats between 0 and 1 referring to 0 and 100 % of the image, respectively.
    The same values are taken for both dimensions of the array.

    :param arr: ndarray
    :param range:
    :return: the maximum z-value,
    """
    min_, max_ = min(range), max(range)
    from_x, to_x = int(min_*arr.shape[0]), int(max_*arr.shape[0])
    from_y, to_y = int(min_ * arr.shape[1]), int(max_ * arr.shape[1])

    mask = np.zeros(arr.shape, np.bool)
    mask[from_x:to_x, from_y: to_y] = 1

    arr = np.where(mask, arr, 0)

    return arr.max(), np.unravel_index(arr.argmax(), arr.shape)


def get_resolution(x: np.ndarray, mean: bool = True) -> float:
    """Get the (mean) resolution of an array.

    This function assumes that the difference for two consecutive points is always
    the same, as it is for a typical image, for instance.

    :param x: The array to get the resolution for
    :param mean: A indicator if the values of the array should be averaged (useful if the steps in x are not constant)
    :return: the difference between two consecutive values
    """

    x_u = np.unique(x[~np.isnan(x)])
    if mean:
        return float(np.mean(x_u[1:] - np.roll(x_u[:-1])))
    return float(x_u[1] - x_u[0])


def angle2D(zero: Point2D, point: Point2D) -> float:
    """Return the angle enclosed by the vector from zero to point and the horizontal.

    zero:   An iterable of length two supplying the xy coordinates of the zero
            point, for instance the center of mass
    point:  The point making up the vector with zero to calculate the angle from
    """

    x_vector = point[0] - zero[0]
    y_vector = point[1] - zero[1]

    cos_alpha = x_vector / (x_vector ** 2 + y_vector ** 2)**0.5

    if y_vector < 0:
        return float(360 - np.rad2deg(np.arccos(cos_alpha)))
    else:
        return float(np.rad2deg(np.arccos(cos_alpha)))


def get_data_for_fit(data: np.array, angles: T.List[float], omit_angle: float, minimum: float = -np.inf,
                     maximum: float = np.inf) -> T.List[np.array]:

    data = crop_data(data, minimum=minimum, maximum=maximum)
    data = remove_angles(data, omit_angle, angles)
    return data


def get_data_for_processing_from_hdf(file: FilePath, crop_to_max: bool = True, crop_to_min: bool = True) -> np.array:
    """Load the data used for further processing.

    This function returns the data that is either inside the roi or the whole transformed data.
    Additionally it can be cropped to either or both the minimum and maximum z-value saved
    in the file.

    :param file: the path to the h5py file containing the data.
    :param crop_to_min: if True the data below the minimum z is removed
    :param crop_to_max: if True the data above the maximum z is removed
    :return: the data set as specified
    """

    from tipeval.config import configuration

    err = IOError(f'Wrong File: Th file {file} does neither contain converted data calculated from a 3D image '
                  'nor does it contain a dataset selected with a ROI using the graphical user interface!')

    with h5py.File(file, 'r') as f:
        key1 = configuration.hdf_keys.crop_data_subgroup
        key2 = configuration.hdf_keys.corner_subgroup
        if key1 not in f.keys() or key2 not in f.keys():
            raise err
        try:
            data = f[configuration.hdf_keys.crop_data_subgroup]['roi_data'][:]
        except KeyError:
            try:
                data = f[configuration.hdf_keys.converted_subgroup]['converted_data'][:]
            except KeyError:
                raise err

        minimum, maximum = f[key1].attrs['lower_limit'], f[key1].attrs['upper_limit']

        if crop_to_max:
            data = crop_data(data, maximum=maximum)
        if crop_to_min:
            data = crop_data(data, minimum=minimum)

    return data


def load_fit_data(file: FilePath) -> np.array:
    """Load the data from an hdf5  file that is going to be used for fitting the tip.

    The data that is used for fitting is determined from the parameters that are saved
    in the hdf5 file. These include:
    - the upper and lower limits in z
    - the region of interest which crops the data
    - the angle that around the tip's edges containing data points that are excluded

    :param file: the path to the hdf5 file
    :return: an array containing the xyz data used for fitting
    """
    from tipeval.config import configuration

    try:
        with h5py.File(file, 'r') as f:
            key1 = configuration.hdf_keys.crop_data_subgroup
            key2 = configuration.hdf_keys.corner_subgroup

            omit_angle = f[key1].attrs['omit_angle']
            corner_angles = f[key2]['coordinates'][:]
    except KeyError:
        raise IOError('File does not contain data that can be fitted!')

    angles = [angle2D(zero=(0, 0), point=phi) for phi in corner_angles]

    data = get_data_for_processing_from_hdf(file)

    data = remove_angles(data, omit_angle, angles)

    return data


def crop_data(data: np.array, minimum: T.Optional[float] = None, maximum: T.Optional[float] = None) -> np.array:
    """Crop an xyz data set to the data points situated within limiting minimum and maximum values (in z)

    :param data: The xyz data set to crop
    :param minimum: the minimum z value that is kept
    :param maximum: the maximum z value that is kept
    :return: the reduced data set (numpy.ndarray with three columns)
    """

    mask_data = data[-1]

    minimum = minimum or -np.inf
    maximum = maximum or np.inf

    with np.errstate(invalid='ignore'):
        mask = (mask_data > minimum) * (mask_data < maximum)

    return np.array([d[mask].flatten() for d in data])


def remove_angles(data: np.array, omit_angle: float, edge_angles: T.List[float]) -> T.List[np.array]:
    """Remove all data points that are within the angular range of the edges +- the omit angles

    Applying this function will result in a list of numpy.ndarrays containing the points
    of the individual tip faces.

    :param data: an xyz dataset containing the points used for fitting
    :param omit_angle: the angular range around the edge angles that will be removed
    :param edge_angles: the angles of the edges of the tip pyramid
    :return: the xyz data sets of the individual faces
    """

    corners = (sorted(edge_angles))

    # we add the omit angle to the angles of the corners and subtract it from the subsequent
    # this gives us the ranges of the data we want to use for each face

    omit_angles: T.List[T.Tuple[float, float]] = []

    for i, _ in enumerate(corners):
        phi1 = corners[i-1] + omit_angle
        phi2 = corners[i] - omit_angle

        phi1 = phi1 - 360 if phi1 > 360 else phi1
        phi2 = phi2 + 360 if phi1 < 0 else phi2

        omit_angles.append((phi1, phi2))

    # convert the x, y coordinates of data to their respective angles
    angles = np.apply_along_axis(lambda x: angle2D((0, 0), x), 0, data[:2])

    data_faces: T.List[np.array] = []

    for phi1, phi2 in omit_angles:
        if phi1 > phi2:  # if phi1 is larger we cross 360° and we need to take values larger than phi1 or smaller phi2
            mask = (angles > phi1) + (angles < phi2)
        else:            # otherwise we need to take values larger than phi1 and smaller than phi2
            mask = (angles > phi1) * (angles < phi2)

        data_faces.append(data[:, mask])

    return data_faces


def get_resource_filename(module: Package, fn: FilePath) -> T.Iterator[Path]:
    """Get the path to a package resource.

    :param module: the module containing the resource files
    :param fn: the file name
    :return: a context manager to the path of the resource
    """
    return path(module, fn)


def angle(v1: np.array, v2: np.array) -> float:
    """Calculate the angle between two vectors v1 and v2."""
    return float(np.rad2deg(np.arccos(np.dot(v1, v2)/np.sqrt(np.dot(v1, v1) * np.dot(v2, v2)))))


def point_in_polygon(point: Point2D, x_polygon: np.array, y_polygon: np.array) -> bool:
    """Determine whether a point is within a polygon using a winding number algorithm.

    See the winding_number function for more details.

    :param point: the x and y coordinates of the point
    :param x_polygon: the x coordinates of the polygon
    :param y_polygon: the y coordinates of the polygon
    :return: True if the point is inside the polygon
    """

    return bool(winding_number(point, x_polygon, y_polygon))


def winding_number(point: Point2D, x_polygon: np.array, y_polygon: np.array) -> float:
    """Determine the winding number of a polygon around a point.

    A winding number of != 0 means that the point is inside the polygon.
    A negative number means that the polygon winds around the point in
    a clockwise fashion and vice versa. This is a very simple algorithm. The
    algorithm is taken from:
    D.G. Alciatore, R. Miranda, A Winding Number and Point-in-Polygon Algorithm.

    :param point: the x and y coordinates of the point
    :param x_polygon: the x coordinates of the polygon
    :param y_polygon: the y coordinates of the polygon
    :return: True if the point is inside the polygon
    """

    # subtract the point from the polygon
    x_polygon -= point[0]
    y_polygon -= point[1]

    w = 0  # the winding number

    for i, (x, y) in enumerate(zip(x_polygon, y_polygon)):

        x_last, y_last = x_polygon[i-1], y_polygon[i-1]

        if y_last * y < 0:
            r = x_last + y_last*(x-x_last)/(y_last - y)
            if r > 0:
                w += 1
            else:
                w -= 1
        elif y_last == 0 and x_last > 0:
            if y > 0:
                w += 0.5
            else:
                w -= 0.5
        elif y == 0 and x > 0:
            if y_last < 0:
                w += 0.5
            else:
                w -= 0.5
    return w


def calculate_cross_sections(x: np.array, y: np.array, z: np.array,
                             contact_depths: np.array,
                             emit_warning: bool = True) -> T.List[T.Optional[T.Tuple[np.array, np.array]]]:
    """Calculate the cross sections for a given array of distances.

    Uses the scikit-image library, namely the find_countours function from
    the measure module.

    :param x, y, z: the arrays of x, y and z coordinates.
    :param contact_depths: the contact depths for which to determine the cross sections
    :param emit_warning: if True, a warning is printed when no cross section can be determined for a contact depth
    :return: A list of cross sections (x and y coordinates). If a cross section could not be safely
    determined (i.e. if the contour was not closed or did not contain the center point) None is appended instead.
    """

    x_u, y_u = len(np.unique(x)), len(np.unique(y))
    max_index = np.where(z.flatten() == 0)[0][0]
    center = np.unravel_index(max_index, (x_u, y_u))[::-1]

    step_x = (np.max(x) - np.min(x)) / z.shape[0]
    step_y = (np.max(y) - np.min(y)) / z.shape[1]

    cross_sections = []

    for distance in contact_depths:
        contours = measure.find_contours(z, distance)

        contains = False

        for n, contour in enumerate(contours):
            x_p, y_p = contour[:, 1], contour[:, 0]
            # checks if the polygon is closed (last and first point are the same)
            if not np.array_equal(contour[0], contour[-1]):
                continue

            if point_in_polygon(center, x_p, y_p):
                cross_sections.append((np.array(x_p)*step_x, np.array(y_p)*step_y))
                contains = True
        if not contains:
            if emit_warning:
                print(f'Warning: For the distance {distance} no closed contour contains the center point.')
            cross_sections.append(None)
    return cross_sections


def polygon_area(x: np.array, y: np.array) -> float:
    """Calculate the area of a polygon with corners having the coordinates x and y.

    This function uses the shoelace formula.

    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    https://en.wikipedia.org/wiki/Shoelace_formula
    https://www.youtube.com/watch?v=0KjG8Pg6LGk

    :param x: The x coordinates of the points to calculate the area of
    :param y: The y coordinates of the points to calculate the area of
    :return: the area enclosed by the coordinates
    """

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def save_tip_to_hdf(tip: Tip, hdf_file: FilePath) -> T.NoReturn:
    """Save a tip into an hdf5 file.

    The tip will be attached to the file attrs with the key 'tip'.

    Args:
        tip: The instance of Tip to be saved.
        hdf_file: The file to save the tip to. Has to be an hdf5 file.

    Returns:
        NoReturn
    """
    from tipeval.core.tips import Tip
    assert isinstance(tip, Tip), 'The object you want to save is not a Tip.'

    with h5py.File(hdf_file, 'a') as f:
        if 'tip' in f.keys():
            del f['tip']
        tip_string = pickle.dumps(tip, protocol=0)
        f['tip'] = np.void(tip_string)


def load_tip_from_hdf(hdf_file: FilePath) -> Tip:
    """Load a tip from an hdf5 file.

    Args:
        hdf_file: The file to load the tip from. The tip has to be saved as  key 'tip'.

    Returns:
        An instance of Tip.
    """
    with h5py.File(hdf_file, 'r') as f:
        if 'tip' not in f.keys():
            raise AttributeError(f'{hdf_file} does not contain a tip!')
        tip = pickle.loads(np.array(f['tip']))
    return tip


def load_af_Hysitron(file: FilePath) -> dict[str: T.Any]:
    with open(file, 'r') as f:
        print(f.read())


def convert_xyz_to_npy(file: FilePath, output_file: FilePath = None, separator: str = '\t',
                       skiprows: int = 0) -> T.NoReturn:
    """Convert an xyz-file to binary numpy file.

    When the text file is converted to a numpy-file loading is much faster the next time. The saved
    file should typically have the .npy-file extension. The date from the saved file can subsequently
    be used together with the ImageData.from_npy() class method.

    :param file: the original xyz-file
    :param output_file: the output file to which the data will be saved. If not supplied the name of
                        the original file will be used with the extension replaced by 'npy'
    :param separator: the separator used in the xyz file
    :param skiprows: the numbers of rows to split in the xyz-file (e.g. the header lines)
    :return: NoReturn
    """
    
    data = np.loadtxt(file, delimiter=separator, skiprows=skiprows, unpack=True)

    if output_file == None:
        file_name, _ = os.path.splitext(file)
        output_file = file_name + '.npy'

    np.save(output_file, data)
