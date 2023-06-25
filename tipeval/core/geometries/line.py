"""
This module contains the line class used for calculation and plotting.
"""

from __future__ import annotations

import typing as T

import mayavi.mlab as mlab
import numpy as np

from tipeval.core.geometries.point import Point
from tipeval.core.typing import RGB


class Line:
    """
    Class representing a line in 3D space. The Line is determined by the parametric
    equation of a line ̱x = ̱p + t*̱v where ̱x is a point on the line, ̱p is the origin
    (basically an arbitrary point on the line), ̱v is a vector of length 3 that
    determines the direction of the line in space and t is an arbitrary float.
    """
    def __init__(self, p: np.array, v: np.array):
        """
        :param p: the origin of the line (an arbitrary point on the line)
        :param v: the direction vector of the line
        """
        self._p = Point(*p)
        self._v = v

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> Line:
        """
        Construct a point from two Point instances (p1 and p2).

        The two points have to both lie on the line.
        :param p1: One point on the line
        :param p2: Another point on the line
        :return: A Line instance
        """

        p1 = p1.to_array()
        p2 = p2.to_array()
        return cls(p1, p2-p1)

    def plot(self, min_z: float, max_z: float, thickness: float = 3, color: RGB = (0, 0, 1),
             show: bool = False, figure = None) -> T.Tuple[Point, Point]:
        """
        Plot the line in a mayavi.mlab figure

        :param min_z: the minimum z-value to which the line should extend
        :param max_z: the maximum z-value to which the line should extend
        :param thickness: the thickness of the displayed line
        :param color: the color of the line
        :param show: if True, mlab.show is called
        :return: the two points between which the line will be plotted
        """

        from tipeval.core.utils.display import plot_between_points
        # determine the two end points of the line to be drawn
        p1 = self.point_on_line(self.get_t_for_z_value(min_z))
        p2 = self.point_on_line(self.get_t_for_z_value(max_z))

        plot_between_points(p1, p2, thickness, color)

        if show:
             mlab.show()

        return p1, p2

    def point_on_line(self, t: float) -> Point:
        """Find point on line at distance t from Line.origin ̱p"""

        return Point(*(self.origin.to_array().flatten() + t*self._v))

    def get_t_for_z_value(self, z: float) -> float:
        """
        Get the corresponding value of t in ̱x = (x, y, z) = ̱p + t*̱v for a given z.

        Where ̱x, ̱p and ̱v are vectors of length 3.

        :param z: the z-value for which to determine t
        :return: t
        """
        return (z-self._p.z)/(self._v[-1])

    @property
    def origin(self) -> Point:
        """The origin ̱p of a line ̱x = ̱p + t*̱v"""
        return self._p

    @property
    def v(self) -> np.array:
        """The direction vector of the line"""
        return self._v

    def __str__(self):
        return f'Line: x = {self._p} + t*{self._v}'
