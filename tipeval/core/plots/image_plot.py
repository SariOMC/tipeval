from typing import Tuple, NoReturn

import h5py
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np

from tipeval.core.plots.base_plot import BasePlot
from tipeval.core.typing import Type, FilePath, ImageLimit
from tipeval.core.utils.data import reshape


class ImagePlot(BasePlot):
    """
    This class is for plotting the data recorded with the imaging method after conversion.

    It uses matplotlib.pyplot.imshow to display the data sets. It is possible to directly supply the x, y and z
    datasets (after reshaping into 2D arrays) or supplying an hdf5 file that has been generated using an instance
    of tipeval.ImageData via ImagePlot.from_hdf.
    """

    from tipeval.config import configuration
    _configuration = configuration

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, *, unit: str = 'nm',
                 figsize: Tuple[float] = (5., 6.), xlim: ImageLimit = None, ylim: ImageLimit = None,
                 output_directory: FilePath = None, **kwargs) -> NoReturn:
        """
        :param x: 2D array of the x-data
        :param y: 2D array of the y-data
        :param z: 2D array of the z-data (the actually displayed image data)
        :param unit: the unit associated with the data (for x and y labels)
        :param figsize: the size of the resulting figure passed to matplotlib.pyplot.figure
        :param xlim: the limit used to display the data in x
        :param ylim: the limit used to display the data in y
        :param output_directory: the directory to which a file will be saved
        :param kwargs: some formatting keywords

            - cmap: a color map, has to be a valid matplotlib color map
            - title: a title used for the figure
        """

        self._x = x
        self._y = y
        self._z = z

        super().__init__(figsize=figsize, output_directory=output_directory)

        self._cmap = kwargs.pop('cmap', None) or self._configuration.plotting.default_color_map
        self._title = kwargs.pop('title', '')

        if kwargs:
            raise TypeError('Unknown arguments supplied to ImagePlot.', list(kwargs.keys()))

        self._unit = unit

        # instantiate the figure
        self._make_figure()

        # when we supply the limits we can directly apply them
        if xlim or ylim:
            self.set_limit(xlim, ylim)

    @classmethod
    def from_hdf(cls: Type, filename: FilePath, **kwargs) -> Type:
        """
        Make an image from an hdf5 file.

        This file has to be made using an instance of tipeval.ImageData

        :param filename: the path to the hdf5 file
        :param kwargs: all keyword arguments are passe to __init__
        :return: an instance of ImagePlot
        """
        with h5py.File(filename, 'r') as f:
            data = f[cls._configuration.hdf_keys["converted_subgroup"]]['converted_data'][:]
            unit = f[cls._configuration.hdf_keys["converted_subgroup"]].attrs['unit']

            if 'title' not in kwargs:
                kwargs['title'] = f.attrs['original_file']

        return cls(*data, unit=unit, **kwargs)

    def _make_figure(self):
        """Display the initial data set."""

        self._ax = self._fig.add_subplot()

        # it is important to set the maximum of y before the minimum as the origin of imshow is in the top left
        self._im = self._ax.imshow(self._z, cmap=self._cmap, extent=(np.nanmin(self._x),
                                                                     np.nanmax(self._x),
                                                                     np.nanmax(self._y),
                                                                     np.nanmin(self._y)))

        self._ax.set_xlabel(f'x ({self._unit})')
        self._ax.set_ylabel(f'y ({self._unit})')
        self._ax.set_title(self._title)
        self._fig.tight_layout()

    def set_limit(self, xlim: ImageLimit = None, ylim: ImageLimit = None) -> NoReturn:
        """
        Apply a limit to the displayed image.

        :param xlim: The limits in x. Can be either a tuple of length 2 specifying the limiting values (in data
                      coordinates) or a single value which is applied symmetrically from the zero point
        :param ylim: analogous to xlim but for the y direction
        :return: NoReturn
        """

        if isinstance(ylim, (float, int)):
            ylim = (-ylim, ylim)

        if isinstance(xlim, (float, int)):
            xlim = (-xlim, xlim)

        if xlim is not None:
            mask_x = (self._x >= min(xlim)) & (self._x <= max(xlim))
        else:
            mask_x = np.ones_like(self._x, dtype=bool)

        x, y, z = self._x[mask_x], self._y[mask_x], self._z[mask_x]

        if ylim is not None:
            mask_y = (y >= min(ylim)) & (y <= max(ylim))
            x, y, z = x[mask_y], y[mask_y], z[mask_y]

        *_, z = reshape(x, y, z)

        xlim = (min(x), max(x))
        ylim = (max(y), min(y))

        self._im.set_data([[]])
        self._im = self._ax.imshow(z, cmap=self._cmap, extent=(*xlim, *ylim))

        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)
        self._fig.tight_layout()

    @property
    def title(self) -> str:
        """The image title. Any string can be set as the title"""
        return self._title

    @title.setter
    def title(self, new_title: str) -> NoReturn:
        self._ax.set_title(str(new_title))
        self._fig.tight_layout()
        self._fig.canvas.draw()

    @property
    def figure(self) -> plt.Figure:
        """The matplotlib figure used."""
        return self._fig

    @property
    def ax(self) -> plt.Axes:
        """The axis object containing the graph,"""
        return self._ax

    @property
    def image(self) -> matplotlib.image.AxesImage:
        """The matplotlib image displayed."""
        return self._im

    def save(self, **kwargs):
        """
        Save the image.

        :param kwargs: all keyword aruments are passed to BasePlot.save.
        :return:
        """
        image_identifier = kwargs.pop('image_identifier', 'original')
        super().save(image_identifier=image_identifier, **kwargs)
