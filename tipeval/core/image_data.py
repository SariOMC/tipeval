"""
This module contains the ImageData class, a class for loading images and converting the image data as well as saving
the data to an hdf5 file.
"""

from datetime import datetime
from functools import lru_cache
import os
import typing as T

import h5py
import numpy as np

from tipeval.core.utils.data import reshape, find_max_image, conversion_factor
from tipeval.core.typing import FilePath, Type


class ImageData:
    """
    A class used for loading and saving the image data recorded by the 3D imaging method.

    This class is mainly a constructor and wrapper for an hdf5 file. This file contains all the data
    and metadata associated with a measurement.
    """

    def __init__(self, data: np.ndarray, method: str, *, unit: str = 'm', output_file: T.Optional[FilePath] = None,
                 autosave: bool = True, **kwargs) -> None:
        """
        :param data: an array of flattened x, y, z data
        :param method: the method used to record the data
        :param output_file: a file path to a file to which the data will be saved
        :param autosave: a flag indicating whether the original data should be saved in the hdf5 file
        :param kwargs: other key word arguments are mainly used as metadata saved in the hdf5 file these include
                - file: type str, the original file name
                - analyst: type str, the name of the person analyzing the data
                - date: datetime.datetime: the date and time the image has been recorded
        """

        from tipeval.config import configuration

        self._configuration = configuration

        self._method = method

        # some metadata supplied as arguments
        self._original_file = kwargs.pop('file', None)
        self._operator = kwargs.pop('analyst', 'unknown')
        self._evaluation_date: datetime = kwargs.pop('date', 'unknown')

        if kwargs:
            raise TypeError('Unknown arguments supplied to ImageData.', list(kwargs.keys()))

        self._original_unit = unit

        # this is going to be the unit of the converted data
        self._unit = ''

        # the handle of the hdf5 file which is going to contain the data
        self.handle: h5py.File

        # reshape the flattened array into the image dimensions
        x, y, z = reshape(*data)

        self._data = np.array([x, y, z])

        self._output_file = self._make_output_file(output_file)

        if autosave:
            self.save_to_hdf()

        # a flag indicating if the data has been converted
        self._data_converted = False
        self._is_closed = False

        self.handle.attrs['method'] = self._method
        self.handle.attrs['original_unit'] = self._original_unit

        self._evaluation_date = datetime.now()
        self.handle.attrs['evaluation_date'] = self._evaluation_date.strftime(self._configuration.date_time_format)

    def save_to_hdf(self) -> T.NoReturn:
        """Save the original data to the hdf5 file"""

        self.handle['original_data'] = self._data
        _, file = os.path.split(self._original_file)
        self.handle.attrs['original_file'] = file

    def _make_output_file(self, file_location: T.Optional[FilePath]) -> FilePath:
        """Generates the output file and sets it as self.handle"""

        if file_location is None:
            output_folder = self._configuration.output_settings.default_output_folder
            output_file = self._configuration.output_settings.default_output_file
        else:
            output_folder, output_file = os.path.split(file_location)

        if not os.path.exists(output_folder) and output_folder != '':
            try:
                os.mkdir(output_folder)
            except FileNotFoundError:
                raise FileNotFoundError('Cannot make folder. Please make sure that the supplied output folder exists!')

        output_file = os.path.join(output_folder, output_file)

        self.handle = h5py.File(output_file, 'w')

        return output_file

    def close(self) -> T.NoReturn:
        """
        Close the hdf5 file handle.

        :return: NoReturn
        """

        if self.handle is not None:
            self.handle.close()
            self._is_closed = True

    def open(self) -> T.NoReturn:
        """Open the hdf5 file if it is closed."""

        if self._is_closed:
            self.handle = h5py.File(self._output_file, 'a')
            self._is_closed = False

    @classmethod
    def from_txt(cls: Type, file_name: FilePath, method: str, separator: str = '\t', skiprows: int = 0,
                 **kwargs) -> Type:
        """
        Make an instance of ImageData from a text file.

        The text file is supposed to contain flattened x, y, z data as three columns.

        :param file_name: the path to the text file
        :param method: the method used to record the data set (typically AFM, CLSM, self-imaging, ...)
        :param separator: the separator used to separate the data in one line
        :param skiprows: the number of header lines in the file
        :param kwargs: the kwargs are passed to ImageData.__init__
        :return: an instance of ImageData
        """

        data = np.loadtxt(file_name, delimiter=separator, skiprows=skiprows, unpack=True)
        return cls(data, method, file=file_name, **kwargs)

    @classmethod
    def from_npy(cls: Type, file_name: FilePath, method: str, **kwargs) -> Type:
        data = np.load(file_name)
        return cls(data, method, file=file_name, **kwargs)

    def convert_data(self, new_unit: str = 'nm', range: T.Tuple[float] = (0., 1.)) -> T.NoReturn:
        """
        Convert the xyz dataset

        Convert the data set such that:
            - 1. The maximum point is in the center of the image (i.e. x and y are shifted)
            - 2. The maximum value is 0 and the depth is counted positively from the top downwards

        The converted data set can be found in the hdf5 file in the group <CONVERTED_GROUP> that is defined in the
        module.

        :param new_unit: the new unit to convert the data to
        :param range: a tuple of length 2 which defines a range wherein the maximum should be searched. Especially
                      with self-imaging it might happen that the maximum is not in the center of the image but somewhere
                      at the edge of the image
        :return: NoReturn
        """

        try:
            factor = conversion_factor(self._original_unit, new_unit)
        except KeyError:
            raise KeyError("Do not recognize units. Only those units saved in the FROM_UNIT and TO_UNIT dictionaries"
                           "can be used.")

        self.open()

        if self.handle.get(f'{self._configuration.hdf_keys["converted_subgroup"]}/x_converted') is not None:
            raise ValueError('Data of the file has already been converted!')

        x, y, z = self._data * factor

        max_z, (index_max_x, index_max_y) = find_max_image(z, range)

        maximum_x, maximum_y = x[index_max_x, index_max_y], y[index_max_x, index_max_y]

        z_conv = (z - max_z) * -1
        x_conv = x-maximum_x
        y_conv = y-maximum_y

        self.handle.attrs['x_at_maximum_z'] = maximum_x
        self.handle.attrs['y_at_maximum_z'] = maximum_y
        self.handle.attrs['maximum_z'] = max_z

        converted_subgroup = self.handle.create_group(self._configuration.hdf_keys["converted_subgroup"])
        converted_subgroup['converted_data'] = np.array([x_conv, y_conv, z_conv])

        converted_subgroup.attrs['proportionality_factor'] = factor

        converted_subgroup.attrs['maximum_at_x_index'] = index_max_x
        converted_subgroup.attrs['x_at_maximum_z'] = maximum_x
        converted_subgroup.attrs['y_at_maximum_z'] = maximum_y
        converted_subgroup.attrs['unit'] = new_unit

        self._unit = new_unit

        self._data_converted = True

    def limit_z(self, limit: float, safety_margin: float = 0.01) -> T.NoReturn:
        """Crop the image data up to a maximum depth.

        The complete cross section at the depth is preserved. This procedure overwrites the
        converted data in the hdf5 file.

        :param limit: the maximum depth that has to be completely preserved.
        :param safety_margin: this safety margin fraction is added to the limit.
        :return: NoReturn
        """

        assert self._data_converted, 'You have to convert the data set first using ImageData.convert_data()'

        max_z = self.handle.attrs['maximum_z']

        assert 0 < limit < max_z, f'The value of the limit must be between 0 and the maximum depth value of ' \
                                  f'the data ({max_z}).'

        x, y, z = self.handle['data_converted/converted_data']

        mask1 = z <= limit * (1 + safety_margin)
        x_new = x[mask1]
        y_new = y[mask1]

        x_min, x_max, y_min, y_max = x_new.min(), x_new.max(), y_new.min(), y_new.max()

        mask2 = (x <= x_max) * (x >= x_min) * (y <= y_max) * (y >= y_min)

        x_crop = x[mask2]
        y_crop = y[mask2]
        z_crop = z[mask2]

        x_crop, y_crop, z_crop = reshape(x_crop, y_crop, z_crop)
        del self.handle['data_converted/converted_data']
        self.handle['data_converted/converted_data'] = np.array([x_crop, y_crop, z_crop])

    @property
    def operator(self) -> str:
        """The person evaluating the data"""
        return self._operator

    @operator.setter
    def operator(self, operator: str) -> T.NoReturn:
        self._operator = str(operator)

    @property
    def original_file(self) -> FilePath:
        """The name of the original data file"""
        return self._original_file

    @property
    def output_file(self) -> FilePath:
        """The output hdf5 file containing all data"""
        return self._output_file

    @property
    @lru_cache()
    def method(self) -> str:
        """The method used for recording the dataset"""
        return self._method

    @property
    def shape(self) -> T.Tuple[int]:
        """The image dimensions"""
        return self.x.shape

    @property
    def x(self) -> np.ndarray:
        """The original x data"""
        return self._data[0]

    @property
    def y(self) -> np.ndarray:
        """The original y data"""
        return self._data[1]

    @property
    def z(self) -> np.ndarray:
        """The original z data"""
        return self._data[2]

    @property
    def data(self) -> np.ndarray:
        """An array containing the original x, y and z"""
        return self._data

    @property
    def is_closed(self):
        """A flag indicating whether the associated hdf5 file has been closed."""
        return self._is_closed

    @property
    def original_unit(self) -> str:
        """The unit of the original data."""
        return self._original_unit

    @property
    def unit(self):
        """The unit of the converted image data."""
        return self._unit

    @property
    def converted_data(self) -> np.array:
        """The image data with the center shifted to the maximum value and reversed z"""

        if not self._data_converted:
            raise AttributeError('The image data has not yet been converted. Call self.convert to generate the '
                                 'converted data.')
        self.open()
        return np.array(self.handle[f'{self._configuration.hdf_keys["converted_subgroup"]}/converted_data'])
