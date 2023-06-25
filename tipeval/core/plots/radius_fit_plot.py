import typing as T

import mayavi.mlab as mlab
import numpy as np

from tipeval.core.geometries.sphere import equation_sphere
from tipeval.core.plots.base_plot import BasePlot


class RadiusFitPlot(BasePlot):
    """
    Class for plotting the fit of a sphere to a set of data points
    """

    def __init__(self, x: np.array, y: np.array, z: np.array, center_coordinates: T.List[float], radius_fit: float,
                 nr_points: int = 20, point_size: float = 3, color: T.Union[str, T.Tuple[float, ...]] = (0, 0, 0),
                 **kwargs) -> T.NoReturn:
        """
        :param x, y, z: The coordinates of the data points used for fitting
        :param center_coordinates: the coordinates (x, y, z) of the center of the fitted sphere
        :param radius_fit: the radius of the sphere
        :param nr_points: the number of points in x and y used for the meshgrid to calculate the surface of the sphere
        :param point_size: the point size used for plotting the data points
        :param color: the color used for the points
        :param kwargs: are passed to mayavi.mlab.points3d
        """

        super().__init__(type='mayavi')

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        kwargs['scale_factor'] = point_size
        kwargs['color'] = color
        nodes = mlab.points3d(x, y, -z, -z, **kwargs)
        nodes.glyph.scale_mode = 'scale_by_vector'

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nr_points), np.linspace(y_min, y_max, nr_points))
        zz = equation_sphere((xx, yy), *center_coordinates, radius_fit)
        mlab.mesh(xx, yy, -zz, color=(0, 0, 1), opacity=0.5)

        mlab.show()
