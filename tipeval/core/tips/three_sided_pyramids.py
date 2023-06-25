"""
This module contains the classes used for fitting a three-sided tip to a data set.

The basic class ThreeSidedPyramidTip can be used for any given cone angle. However, this module also
contains some convenience classes for Berkovich and Cube-corner tips.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import zip_longest
import typing as T

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from tipeval.core.tips.tip import Tip
from tipeval.core.Errors import CalculationError
from tipeval.core.geometries import Line, Plane, Point
from tipeval.core.geometries.sphere import equation_sphere
from tipeval.core.plots import FitDataPlot, RadiusFitPlot
from tipeval.core.typing import FilePath, Depth, RGB, Figure
from tipeval.core.utils.data import load_fit_data, angle, calculate_cross_sections, polygon_area, crop_data
from tipeval.core.utils.nanoindentation import (area_function_polynomial, area_function_polynomial_fixed,
                                                find_zero, contact_radius_sphere, contact_radius_cone,
                                                OPENING_ANGLE_BERKOVICH, OPENING_ANGLE_CUBE_CORNER, area_cone,
                                                area_function_hysitron, first_coefficient_polynomial,
                                                area_function_umis, IDEAL_ANGLES)


class ThreeSidedPyramidTip(Tip):
    """Class representing three sided pyramid.

    Can be used for Berkovich and cube corner tips as well but for these separate classes exist.
    The class is constructed from three Plane instances. By using the classmethod 'from_file' these planes
    can be automatically fitted to data saved in a file.
    """
    def __init__(self, ideal_angle: float, type: str, planes: T.List[Plane], file: FilePath = None,
                 unit: str = '-', name: str = '') -> T.NoReturn:
        """
        :param ideal_angle: the angle in degrees that the tip should ideally have.
        :param type: the type of the tip. Can be any string. For
        :param planes: a list of three planes making up the pyramid
        :param file: the file storing the data associated with the tip. Has to be supplied in case that the
        area function should be evaluated.
        :param unit: the unit of the supplied data
        :param name: an identifier for the tip
        """

        self._type = type
        self._ideal_angle = ideal_angle
        self._faces = planes
        self._file = file

        self._unit = unit

        self._name = name or 'no name'

        self.contact_depths: T.Optional[np.ndarray] = None
        self.cross_sections: T.List[T.Tuple[np.array, np.array]] = []
        self.areas: T.Optional[np.ndarray] = None
        self.area_ratios: T.Optional[np.ndarray] = None

        self.fit_coefficients: T.List[float] = []
        self.error_fit_coefficients: T.List[float] = []

        self.ideal_coefficients = [first_coefficient_polynomial(self._ideal_angle)]

        self._tip_radius_fit: T.Optional[float] = None
        self._error_tip_radius_fit: T.Optional[float] = None

        self._tip_radius_area_function: T.Optional[float] = None

        self._radius_fit_contact_radius: T.Optional[float] = None
        self._error_radius_fit_contact_radius: T.Optional[float] = None
        self._equivalent_cone_angle_fit_contact_radius: T.Optional[float] = None
        self._error_equivalent_cone_angle_fit_contact_radius: T.Optional[float] = None

    def plot_fit(self, data: T.Iterable[T.Iterable[np.array, np.array, np.array], ...], color_planes: T.Iterable[RGB] = (),
                 opacity_planes: float = 0.25, save_fig: bool = False, show: bool = True) -> FitDataPlot:
        """Show the fit in a 3D mayavi plot

        :param data: the data sets to show for each of the planes. This should be an interable of x, y, z
        coordinates.
        :param color_planes: a sequence of RGB color values with one for each plane. If the sequence is too
        short the last value will be taken for the remaining planes.
        :param opacity_planes: determines the transparency of the fitted planes. A value of 0 would
        mean completely transparent while 1 means completely opaque.
        :param save_fig: if True, the figure will be saved in the default output folder
        :param show:
        :return:
        """

        from tipeval.config.configuration import configuration

        plot = FitDataPlot(data)

        mins = plot.minimums
        maxs = plot.maximums

        colors = color_planes or configuration.plotting.colors_fit_faces

        assert len(colors) <= len(self._faces), f'The number of supplied colors must be shorter than or' \
                                                f'equal to the number of planes ({len(self._faces)} in this case).'
        try:
            len(colors[0])
        except TypeError:
            raise AssertionError('Please ensure that the colors are supplied as a Tuple of Tuples with length 3. '
                                 'A single color has to be supplied in the shape of ((1, 0, 0),).')

        for plane, color in zip_longest(self._faces, colors, fillvalue=colors[-1]):
            plane.plot_mayavi(mins, maxs, color=color, opacity=opacity_planes, figure=plot.figure)

        self.apex.plot(figure=plot.figure)
        self.plot_axis(maxs[-1], figure=plot.figure)
        self.plot_frame(maxs[-1], figure=plot.figure)

        if save_fig:
            plot.save(image_identifier='fit_data')

        if show:
            plot.show()

        return plot

    def plot_fit_from_file(self, color_planes: T.Iterable[RGB] = (), opacity_planes: float = 0.25,
                           save_fig: bool = False) -> T.NoReturn:
        """Plot the fit of the tip.

        The datasets used for fitting the planes of the tip together with the
        fitted planes are displayed. This method uses tipeval.FitDataPlot together with
        the plot method of tipeval.Plane.

        :param color_planes: The color used for plotting the displayed planes. Must be supplied as a Tuple of
        tuples of length 3. A single color for all planes has to be supplied as ((1, 0, 0),) for instance.
        :param opacity_planes: The opacity of the displayed planes
        :param save_fig: if True, the plot is saved.
        :return: NoReturn
        """

        data = load_fit_data(self._file)
        self.plot_fit(data, color_planes, opacity_planes, save_fig, show=True)

    @classmethod
    def from_file(cls, file: FilePath, ideal_angle: float, type: str, name: str = '') -> ThreeSidedPyramidTip:
        """Create an instance of ThreeSidedPyramidTip from a dataset saved in an hdf5 file.

        :param file: The path to the file containing the dataset.
        :param ideal_angle: the ideal opening half angle of the tip
        :param type: the type of the tip, e.g. Berkovich, Cube-corner...
        :return: An instance of a ThreeSidedPyramid.
        """

        data = load_fit_data(file)
        planes = [Plane.from_dataset(*face) for face in data]

        from tipeval.config import configuration

        with h5py.File(file, 'r') as f:
            unit = f[configuration.hdf_keys["converted_subgroup"]].attrs['unit']

        return cls(ideal_angle=ideal_angle, type=type, planes=planes, file=file, unit=unit, name=name)

    @property
    def faces(self):
        return self._faces

    @property
    def ideal_angle(self) -> float:
        """The theoretical angle of the tip."""
        return self._ideal_angle

    @property
    def type(self) -> str:
        """The type of the tip, e.g. Berkovich."""
        return self._type

    @property
    def unit(self) -> str:
        """The unit of the data points used for fitting."""
        return self._unit

    @property
    def name(self) -> str:
        """The name of the tip"""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = str(name)

    @property
    @lru_cache()
    def apex(self) -> Point:
        """A Point representing the intersection of the three planes"""
        apex = Plane.intersect_with_two_planes(*self._faces)
        return apex

    @property
    @lru_cache()
    def axis(self) -> Line:
        """A Line representing the axis of the tip"""
        p = self.apex.to_array()
        for face in self._faces:
            p += face.normal_vector

        return Line.from_points(self.apex, Point(*p))

    @property
    @lru_cache()
    def equivalent_cone_angle(self) -> float:
        """The opening half angle of a cone with equivalent area to depth ratio"""
        return np.rad2deg(np.arctan(np.sqrt(3*np.sqrt(3)/np.pi)*np.tan(self.tip_angle*np.pi/180)))

    @property
    def equivalent_cone_angle_fit_contact_radius(self) -> float:
        """The opening half angle of a cone with equivalent area to depth ratio determined from the contact radius"""
        return self._equivalent_cone_angle_fit_contact_radius

    def alpha(self) -> float:
        """Alternative to 'equivalent_cone_angle'"""
        return self.equivalent_cone_angle

    @property
    @lru_cache()
    def axis_inclination_angle(self) -> float:
        """The inclination angle of the tip axis with the z-direction in °"""
        inc_angle = abs(angle(self.axis.v, np.array([0, 0, -1])))
        inc_angle = 180 - inc_angle if inc_angle > 90 else inc_angle
        return inc_angle

    @property
    @lru_cache()
    def angles_faces(self) -> T.List[float]:
        """The angles between the tip axis and the individual faces of the tip"""
        return [90 - abs(angle(self.axis.v, face.normal_vector)) for face in self._faces]

    @property
    @lru_cache()
    def tip_angle(self) -> float:
        """The mean of the angles enclosed by the faces and the tip axis"""
        return float(np.mean(self.angles_faces))

    def plot_frame(self, max_z: float, thickness: T.Optional[float] = None, color: T.Optional[RGB] = None, **kwargs) -> T.NoReturn:
        """Plot the frame of the fitted tip

        :param max_z: the maximum z value up to which the frame should be plotted
        :param thickness: the thickness of the frame
        :param color: the color of the frame
        :param kwargs: keyword arguments passed to mlab.plot3d
        :return:
        """

        from tipeval.core.utils.display import plot_between_points
        from tipeval.config import configuration

        color = color or configuration.plotting.color_frame
        thickness = thickness or configuration.plotting.thickness_line_plots

        points: T.List[Point] = []

        for edge in self.edges:
            p1, p2 = edge.plot(self.apex.z, max_z, thickness, color)
            # determine which of the two points is not the apex and append it
            p = p1 if abs(p1.z-self.apex.z) > abs(p2.z-self.apex.z) else p2
            points.append(p)
            plot_between_points(p, self.apex, thickness, color)

        for i, point in enumerate(points):
            plot_between_points(point, points[i-1], thickness, color, **kwargs)

    def plot_axis(self, max_z: float, thickness: T.Optional[float] = None, color: T.Optional[RGB] = None,
                  figure: Figure = None) -> T.NoReturn:
        """Plot the fitted axis of the tip

        :param max_z: the maximum z value up to which the axis should be drawn
        :param thickness: the thickness of the axis
        :param color: the color used for the axis
        :return: NoReturn
        """

        from tipeval.config import configuration

        color = color or configuration.plotting.axis_color
        thickness = thickness or configuration.plotting.axis_thickness

        self.axis.plot(self.apex.z, max_z, thickness, color, figure=figure)

    @property
    @lru_cache()
    def edges(self) -> T.Tuple[Line]:
        """Line objects representing the edges of the fitted tip"""
        return tuple([face.intersect(self._faces[i-1]) for i, face in enumerate(self._faces)])

    def calculate_cross_sections(self, distances: Depth, plot: bool = False,
                                 cmap: T.Optional[str] = None, show_legend: bool = False,
                                 linewidth: float = 1, save_plot: bool = True) -> T.NoReturn:
        """Calculate the cross sections of the tip data for a given list of distances.

        The cross sections are found using scikit-image. Only those cross sections that are closed and contain
        the central point of the tip are considered.

        :param distances: a list (ndarray) of values for which to determine the cross sections
        :param plot: if True a plot of the cross sections is produced
        :param cmap: the color map to use for plotting (a valid matplotlib identifier string)
        :param show_legend: if True the legend will be shown. If the number of depths is quite large
        there might not be enough room for the legend
        :param linewidth: the linewidth used to draw the cross sections
        :param save_plot: if True, the plot is going to be saved.
        :return: the cross sections
        """

        from tipeval.config import configuration

        with h5py.File(self._file, 'r') as f:
            data = np.array(f[configuration.hdf_keys.converted_subgroup]['converted_data'])

            # here we crop the data to the upper limit by setting all values above to limit
            # to the limit itself (+ 1% to be able to pass exactly the value)
            try:
                limit = f[configuration.hdf_keys.crop_data_subgroup].attrs['upper_limit']
            except KeyError:
                pass
            else:
                data[-1][data[-1] > limit] = limit * 1.01

        cross_sections = calculate_cross_sections(*data, distances)

        values = [(distance, cross_section) for distance, cross_section in zip(distances, cross_sections)
                  if cross_section is not None]

        self.contact_depths = np.array([value[0] for value in values])
        self.cross_sections = [value[1] for value in values]

        self.areas = np.array([polygon_area(*cross_section) for cross_section in self.cross_sections])
        self.area_ratios = self.areas/area_function_polynomial(self.contact_depths, *self.ideal_coefficients)

        if plot or save_plot:
            self._plot_cross_sections(show_legend, cmap, linewidth, save_plot)

    def _plot_cross_sections(self, show_legend: bool, cmap: str = None, linewidth: float = 1,
                             save_plot: bool = False) -> T.NoReturn:
        """Plot the cross sections.

        :param show_legend: if True the legend is shown.
        :param cmap: the color map to use for plotting (a valid matplotlib identifier string)
        :param linewidth: the linewidth used to draw the cross sections
        :param save_plot: if True, the image is going to be saved
        :return: NoReturn
        """

        from tipeval.config import configuration
        from tipeval.core.plots import CrossSectionsPlot

        cmap = cmap or configuration.plotting.color_map_cross_sections
        cmap = plt.get_cmap(cmap)
        colors = [cmap(x) for x in np.linspace(0, 1, len(self.contact_depths))]

        plot = CrossSectionsPlot(self.contact_depths, self.cross_sections, colors, self._unit, linewidth)
        plot.show(show_legend)

        if save_plot:
            plot.save(image_identifier='cross_sections')

    def fit_area_function(self, coefficients: T.Union[T.List[float], int] = 5, fix_first_coeff: bool = False,
                          plot: bool = False, plot_ratio: bool = False, save_plot: bool = True) -> T.NoReturn:
        """Fit the area function to the measured cross sections.

        :param coefficients: can be either an integer specifying the number of coefficients used or an actual list
                             of inital guesses for the coefficients
        :param fix_first_coeff: if True then the first value is either fixed to the ideal one if only a number is
                                given or to the first value in the supplied list.
        :param plot: if True the fit is going to be plotted
        :param plot_ratio: if True then actually the ratio of the measured area with the ideal area is plotted
                           instead of the measured area
        :return: NoReturn
        """

        if not self.cross_sections:
            raise AttributeError('No cross sections were yet calculated!')

        if isinstance(coefficients, int):
            first_coefficient = area_cone(1, self._ideal_angle)
            coefficients = [first_coefficient] + [1] * (coefficients - 1)

        assert len(coefficients) <= 6, 'The maximum number of allowed coefficients is 6.'

        # we weigh the area by the area values such that the fit at lower contact depths is better
        if fix_first_coeff:
            ans, err = curve_fit(lambda x, *c: area_function_polynomial_fixed(x, *c, coefficient0=coefficients[0]),
                                 self.contact_depths, self.areas, coefficients[1:], sigma=self.areas)
            ans = [coefficients[0]] + list(ans)
            err = [0] + list(np.sqrt(np.diag(err)))
        else:
            ans, err = curve_fit(area_function_polynomial, self.contact_depths, self.areas, coefficients, sigma=self.areas)
            ans = list(ans)
            err = np.sqrt(np.diag(err))

        self.fit_coefficients = ans + [0] * (6-len(ans))
        self.error_fit_coefficients = err + [0] * (6-len(err))

        if plot or save_plot:
            self.plot_area_fit(plot_ratio, save_plot)

    def plot_area_fit(self, plot_ratio: bool, save_plot: bool, show: bool = False, **kwargs) -> T.NoReturn:
        """Plot the fit of the cross section areas

        :param plot_ratio: if True then actually the ratio of the measured area with the ideal area is plotted
        instead of the measured area
        :param kwargs: keyword arguments are passed to AreaFitPlot
        :return: NoReturn
        """

        from tipeval.core.plots import AreaFitPlot
        plot = AreaFitPlot(self.contact_depths, self.areas, self.fit_coefficients, plot_ratio, self.ideal_coefficients,
                           self._unit)
        if show:
            plot.show()
        if save_plot:
            plot.save(image_identifier='area_function')

    def fit_info(self, widths: T.Tuple[int, int, int] = (33, 10, 15)):
        """Make a table with some information of the fit of the tip

        :param widths: the widths of the individual columns
        :return: the printed information
        """

        ideal_angle, cone_angle = IDEAL_ANGLES[self.type]

        text = f"""{'Method': ^{widths[0]}}|{'Value':^{widths[1]}}|{'Ideal value':^{widths[2]}}
{'|'.join(['+' * val for val in widths])}
{'Tip axis inclination angle':^{widths[0]}}|{self.axis_inclination_angle:^{widths[1]}.1f}|{0:^{widths[2]}.1f}
{'Angle between face/axis':^{widths[0]}}|{self.angles_faces[0]:^{widths[1]}.1f}|{ideal_angle:^{widths[2]}.1f}
{'':^{widths[0]}}|{self.angles_faces[1]:^{widths[1]}.1f}|{ideal_angle:^{widths[2]}.1f}
{'':^{widths[0]}}|{self.angles_faces[2]:^{widths[1]}.1f}|{ideal_angle:^{widths[2]}.1f}
{'Tip angle (average face angle)':^{widths[0]}}|{self.tip_angle:^{widths[1]}.1f}|{ideal_angle:^{widths[2]}.1f}
{'Cone angle pyramid fit':^{widths[0]}}|{self.equivalent_cone_angle:^{widths[1]}.1f}|{cone_angle:^{widths[2]}.1f}
{'Cone angle contact radius':^{widths[0]}}|{self._equivalent_cone_angle_fit_contact_radius:^{widths[1]}.1f}|{cone_angle:^{widths[2]}.1f}"""
        return text

    def _factor_sphere_transition(self):
        """The ratio of area to ideal area where the sphere transitions to a cone."""
        return (1 / np.sin(np.deg2rad(self.equivalent_cone_angle)) + 1) ** 2

    def depth_sphere_transition(self, radius_guess: float = 100) -> float:
        """Get the depth from the area function where the tip transitions from a sphere to a pyramid (cone).

        This depth is needed to determine the radius from the area function and can be used as
        a useful limit for fitting a sphere to the data points (also for determining the tip radius).

        :param radius_guess: an initial guess of the tip radius.
        :return: the depth value
        """
        if not self.fit_coefficients:
            raise CalculationError('You have to fit the area function first! Use the method fit_area_function.')

        factor = self._factor_sphere_transition()

        hs_guess = radius_guess * (1 - np.sin(np.deg2rad(self.equivalent_cone_angle)))

        hs = find_zero(lambda x: -factor + area_function_polynomial(x, *self.fit_coefficients)
                                 / area_function_polynomial(x, *self.ideal_coefficients),
                       hs_guess)
        return hs

    def radius_from_area_function(self, radius_guess: float = 100, plot: bool = True, save_plot: bool = False) -> float:
        """The tip radius determined from the area function.

        The tip radius is calculated using the Newton method from the area ratio. The
        procedure can be deduced from the Book "Nanoindentation" of A. Fischer-Cripps
        pp. 86-89.

        For a cone with semiangle alpha the spherical tip meets the cone at hs, which
        is
            hs = R(1-sin(alpha)),

        where R is the tip radius. At the same time, the ratio of A/Ai (A...real area,
        Ai...ideal area) is also determined by the cone angle to be:

            A/Ai = 1/sin(alpha + 1)²

        For a Berkovich tip, A/Ai is about 4.25. At this ratio, the contact depth
        equals hs. Thus, by finding this contact depth from the area function (e.g. by
        using the Newton method) it is possible to determine the tip radius of an
        equivalent sphero-conical indenter.

        :param radius_guess: an initial guess for radius
        :param plot: if True, a plot of the evaluation will be shown
        :param save_plot: if True, the plot will be saved
        :return: the thus evaluated radius
        """

        hs = self.depth_sphere_transition(radius_guess=radius_guess)
        factor = self._factor_sphere_transition()

        radius = hs/(1-np.sin(np.deg2rad(self.equivalent_cone_angle)))

        self._tip_radius_area_function = radius

        if plot or save_plot:
            from tipeval.core.plots import RadiusAreaFunctionPlot
            graph = RadiusAreaFunctionPlot(self.contact_depths, self.areas, self.fit_coefficients,
                                           self.ideal_coefficients, hs, factor, unit=self._unit)
            graph.show(legend=True)
            if save_plot:
                graph.save(image_identifier='radius_area_function')

        return radius

    @property
    @lru_cache()
    def blunting_depth(self) -> float:
        """The blunting depth corresponds to the z value of the apex of the fit"""
        return np.abs(self.apex.z)

    @property
    @lru_cache()
    def radius_blunting_depth(self) -> float:
        """The radius calculated from the blunting depth

        The tip radius R can be calculated from the blunting depth hb and the equivalent cone angle
        alpha via:
        R = hb*sin(alpha)/(1-sin(alpha))
        """
        return (self.blunting_depth * np.sin(np.deg2rad(self.equivalent_cone_angle))
                / (1-np.sin(np.deg2rad(self.equivalent_cone_angle))))

    def radius_fit(self, limit: float, plot: bool = True, init_guess: T.List[float] = (0, 0, 400, 400),
                   save_fig: bool = False, **kwargs) -> T.Tuple[float, float]:
        """Fit a sphere to the recorded data points.

        This method determines the tip radius by fitting the surface of a sphere to the top
        part of the recorded data points.

        :param limit: the contact depth up to which the data points are used for the fit.
        :param plot: if True, a plot of the fit is shown
        :param init_guess: an initial guess for the fit parameters. The first three are
        the x, y and z coordinates of the center and the fourth is the radius.
        :param save_fig: if True, the figure is saved to a file.
        :param kwargs: are passed to the RadiusFitPlot class that plots the fit.

        :return: the obtained radius and error
        """

        from tipeval.config import configuration

        with h5py.File(self._file, 'r') as f:
            if 'roi_data' in f[configuration.hdf_keys.crop_data_subgroup].keys():
                data = np.array(f[configuration.hdf_keys.crop_data_subgroup]['roi_data'])
                data = data[:, ~np.isnan(data[0])]
            else:
                data = np.array(f[configuration.hdf_keys.converted_subgroup]['converted_data'])

        x, y, z = crop_data(data, maximum=limit)

        res, matrix = curve_fit(equation_sphere, (x, y), z, init_guess)
        *center_coordinates, radius_fit = res
        errors = np.sqrt(np.diag(matrix))

        if plot or save_fig:
            graph = RadiusFitPlot(x, y, z, center_coordinates, radius_fit, **kwargs)
            graph.save(image_identifier='fit_sphere')

        return radius_fit, errors[-1]

    @property
    def contact_radius(self) -> np.array:
        """The contact radius calculated for the already given contact depths."""
        if self.areas is None:
            raise CalculationError('Please calculate contact areas first by setting a set of '
                                   'contact depths using the calculate_cross_sections method.')

        return np.sqrt(self.areas/np.pi)

    def radius_from_contact_radius(self, limit_radius: float, determine_cone_angle: bool = True,
                                   limit_cone_angle: T.Optional[float] = None, plot: bool = True,
                                   plot_legend: bool = True, initial_guess_radius: float = 100,
                                   initial_guess_cone_angle: T.Tuple[float, float] = (70, -10),
                                   save_plot: bool = True) -> T.Tuple[float, ...]:
        """Determine the tip radius (and the equivalent cone angle) from the contact radius

        The contact radius rc is calculated as:
        rc * sqrt(Ac/π),
        where Ac is the contact area. The tip radius can be determined by fitting a circle to the first
        contact radius values. The equivalent cone angle can be determined by a linear fit of the
        contact radius farther away from the top.

        :param limit_radius: the contact radius limit used for fitting the sphere. Only smaller values are used.
        :param determine_cone_angle: if True the equivalent cone angle is also determined.
        :param limit_cone_angle: the contact radius limit used for fitting the line. Only larger values are used.
        :param plot: if True, the resulting fit is plotted.
        :param plot_legend: if True, a legend is shown
        :param initial_guess_radius: an initial guess for the radius value.
        :param initial_guess_cone_angle: an initial guess for the slope and intercept of the linear fit.
        :return: if the cone angle is not determined, the radius and its error are returned. Otherwise
        :param save_plot: if True, the plot is going to be saved
        these two plus the value for the cone angle and its error are returned.
        """

        limit_cone_angle = limit_cone_angle if limit_cone_angle is None else limit_radius
        mask1 = self.contact_depths < limit_radius
        mask2 = self.contact_depths > limit_cone_angle
        radius_sphere, error_radius_sphere = curve_fit(contact_radius_sphere, self.contact_depths[mask1],
                                                       self.contact_radius[mask1], p0=initial_guess_radius)

        (angle, intercept), error_theta = curve_fit(contact_radius_cone, self.contact_depths[mask2],
                                                    self.contact_radius[mask2], p0=initial_guess_cone_angle)

        if plot or save_plot:
            from tipeval.core.plots import ContactRadiusFitPlot

            plot = ContactRadiusFitPlot(self)
            plot.plot_contact_radius()
            plot.plot_fit_sphere(radius_sphere)

            if determine_cone_angle:
                plot.plot_fit_contact_angle(angle, intercept)

            plot.show(plot_legend)
            if save_plot:
                plot.save(image_identifier='contact_radius')

        self._radius_fit_contact_radius = radius_sphere[0]
        self._error_radius_fit_contact_radius = error_radius_sphere[0, 0]
        self._equivalent_cone_angle_fit_contact_radius = angle
        self._error_equivalent_cone_angle_fit_contact_radius = np.sqrt(np.diag(error_theta)[0])

        if determine_cone_angle:
            return (self._radius_fit_contact_radius, self._error_radius_fit_contact_radius,
                    self._equivalent_cone_angle_fit_contact_radius ,
                    self._error_equivalent_cone_angle_fit_contact_radius)
        return self._radius_fit_contact_radius, self._error_radius_fit_contact_radius

    def save_area_function(self, file_name: FilePath, system: str, compliance: T.Optional[float] = None,
                           method: str = '') -> T.NoReturn:
        """Export the tip information into an area function.

        :param file_name: the file name of the resulting area function.
        :param system: the system for which to save the area function. The implemented
        systems include:
            - Hysitron
            - UMIS
        :param compliance: the elastic compliance of the system (only used with system='UMIS')
        :param method: the method used for recording the image data (only used with system='UMIS')
        :return: NoReturn
        """

        if system == 'Hysitron':
            text = area_function_hysitron(self)
        elif system == 'UMIS':
            text = area_function_umis(self, compliance=compliance, method=method)
        else:
            raise NotImplementedError(f"System '{system}' is not (yet) implemented.")

        with open(file_name, 'w') as f:
            f.write(text)


class BerkovichTip(ThreeSidedPyramidTip):
    """Class representing a Berkovich tip.

    Subclasses ThreeSidedPyramidTip with essentially the same constructor arguments (see __init__ there for
    further information). However, for convenience the ideal_angle and type are automatically filled.
    """
    def __init__(self, planes: T.List[Plane], file: FilePath = None, unit: str = '-', **kwargs):

        kwargs.setdefault('ideal_angle', OPENING_ANGLE_BERKOVICH)
        kwargs.setdefault('type', 'Berkovich')

        super().__init__(planes=planes, file=file, unit=unit, **kwargs)

    @classmethod
    def from_file(cls, file: FilePath, **kwargs) -> ThreeSidedPyramidTip:
        """Overrides the from_file method of ThreeSidedPyramidTip.

        :param file: the file to load the data from.
        :param kwargs: are passed to the constructor of ThreeSidedPyramidTip.
        :return: an instance of BerkovichTip
        """
        kwargs.setdefault('ideal_angle', OPENING_ANGLE_BERKOVICH)
        kwargs.setdefault('type', 'Berkovich')
        return super().from_file(file=file, **kwargs)


class CubeCornerTip(ThreeSidedPyramidTip):
    """Class representing a cube corner tip.

    Subclasses ThreeSidedPyramidTip with essentially the same constructor arguments (see __init__ there for
    further information). However, for convenience the ideal_angle and type are automatically filled.
    """
    def __init__(self, planes: T.List[Plane], file: FilePath = None, unit: str = '-', **kwargs):

        kwargs.setdefault('ideal_angle', OPENING_ANGLE_CUBE_CORNER)
        kwargs.setdefault('type', 'Cube-corner')

        super().__init__(planes=planes, file=file, unit=unit, **kwargs)

    @classmethod
    def from_file(cls, file: FilePath, **kwargs) -> ThreeSidedPyramidTip:
        """Overrides the from_file method of ThreeSidedPyramidTip.

        :param file: the file to load the data from.
        :param kwargs: are passed to the constructor of ThreeSidedPyramidTip.
        :return: an instance of CubeCornerTip
        """

        kwargs.setdefault('ideal_angle', OPENING_ANGLE_CUBE_CORNER)
        kwargs.setdefault('type', 'Cube-corner')
        return super().from_file(file=file, **kwargs)
