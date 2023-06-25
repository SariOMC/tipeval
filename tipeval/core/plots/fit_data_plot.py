"""
This module contains the class for plotting the fit of the pyramid faces to the recorded data points.
"""

import typing as T

import mayavi.mlab as mlab
import numpy as np

from tipeval.core.plots.base_plot import BasePlot
from tipeval.core.typing import FilePath


class FitDataPlot(BasePlot):
    """
    Class for plotting the fit of the faces of a pyramid tip.
    """

    def __init__(self, faces_data: T.Iterable[T.Iterable[np.array]], point_size: float = 10.,
                 output_directory: T.Optional[FilePath] = None, **kwargs) -> T.NoReturn:
        """
        :param faces_data: a list of the x, y and z data sets.
        :param point_size: the size of the displayed data points
        :param kwargs: keyword arguments passed to mlab.points3d
        """

        super().__init__(type='mayavi', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), output_directory=output_directory)

        self._data = faces_data

        # the mimina and maxima of the x, y, z values
        self._minimums, self._maximums = [func(np.hstack(self._data), axis=1) for func in (np.min, np.max)]

        kwargs['scale_factor'] = point_size
        kwargs['colormap'] = kwargs.get('colormap', 'autumn')

        self._plot(**kwargs)

    def _plot(self, **kwargs) -> T.NoReturn:

        for data_set, color in zip(self._data, [(0, 0, 1), (0, 1, 0), (1, 0, 0)]):
            x, y, z = data_set
            nodes = mlab.points3d(x.flatten(), y.flatten(), -z.flatten(), z.flatten(), figure=self._fig, **kwargs)
            nodes.glyph.scale_mode = 'scale_by_vector'   # makes that all points are equally sized

        self._fig.scene.parallel_projection = True

    @property
    def maximums(self) -> T.List[float]:
        """The maximum values of x, y, z"""
        return list(self._maximums)

    @property
    def minimums(self) -> T.List[float]:
        """The minimum values of x, y, z"""
        return list(self._minimums)
