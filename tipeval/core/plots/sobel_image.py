import typing as T

import numpy as np

from tipeval.core.plots.image_plot import ImagePlot
from tipeval.core.typing import ImageLimit
from tipeval.core.utils.display import sobel_filter


class SobelPlot(ImagePlot):
    """
    Plot showing a Sobel filtered version of an image.

    This class is essentially the same as tipevel.ImagePlot. The only difference is that the image data is
    subjected to a Sobel filter before display.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, *, unit: str = 'nm',
                 figsize: T.Tuple[float] = (5., 6.), xlim: ImageLimit = None, ylim: ImageLimit = None,
                 **kwargs) -> None:
        """
        :param x: 2D array of the x-data
        :param y: 2D array of the y-data
        :param z: 2D array of the z-data (the actually displayed image data)
        :param unit: the unit associated with the data (for x and y labels)
        :param figsize: the size of the resulting figure passed to matplotlib.pyplot.figure
        :param xlim: the limit used to display the data in x
        :param ylim: the limit used to display the data in y
        :param kwargs: some formatting keywords are passed to ImagePlot.__init__. But two keywords are passed to
        scipy.ndimage.sobel, namely mode and cval. Look there for the documentation

            - cmap: a color map, has to be a valid matplotlib color map
            - title: a title used for the figure
        """

        mode = kwargs.pop('mode', 'reflect')
        cval = kwargs.pop('cval', 0.0)

        z = sobel_filter(z, mode=mode, cval=cval)

        super().__init__(x, y, z, unit=unit, figsize=figsize, xlim=xlim, ylim=ylim, **kwargs)

    @property
    def filtered_data(self) -> np.array:
        """The image after applying the filter."""
        return self._z

    def save(self, **kwargs) -> T.NoReturn:
        """
        Save the image.

        :param kwargs: all keyword aruments are passed to BasePlot.save.
        :return:
        """
        image_identifier = kwargs.pop('image_identifier', 'sobel')
        super().save(image_identifier=image_identifier, **kwargs)
