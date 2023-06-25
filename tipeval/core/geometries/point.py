import typing as T

import mayavi.mlab as mlab
import numpy as np

from tipeval.core.typing import RGB


class Point:
    def __init__(self, x: float, y: float, z: float) -> T.NoReturn:

        self._x = x
        self._y = y
        self._z = z

    def plot(self, point_size: float = 20., color: RGB = (0., 0., 0.), text_width: float = 0.2, show: bool = False,
             **kwargs) -> T.NoReturn:

        figure = kwargs.pop('figure', None)

        mlab.points3d([self._x], [self._y], [-self._z], color=color, scale_factor=point_size, figure=figure)
        mlab.text(self._x, self._y, 'Apex', z=-self._z, width=text_width, color=color, figure=figure)

        if show:
            mlab.show()

    def to_array(self) -> np.ndarray:
        """Convert the point coordinates to a numpy array of length 3."""
        return np.array(list(self))

    @property
    def x(self) -> float:
        """The point's x coordinate"""
        return self._x

    @property
    def y(self) -> float:
        """The point's y coordinate"""
        return self._y

    @property
    def z(self) -> float:
        """The point's z coordinate"""
        return self._z

    def __iter__(self) -> T.Iterator[float]:
        return iter((self._x, self._y, self._z))

    def __str__(self):
        return f'Point: ({self.x}, {self.y}, {self.z})'