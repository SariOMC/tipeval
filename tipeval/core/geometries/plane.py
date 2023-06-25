"""
This module contains the Plane class and associated utilities.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NoReturn, List

import numpy as np
from mayavi import mlab
from scipy.optimize import curve_fit

from tipeval.core.geometries.point import Point
from tipeval.core.typing import RGB
from tipeval.core.geometries.line import Line

# one value needs to b
PLANE_COEFF_FIXED = 1.


def equation_plane(X, a, b, d, c=PLANE_COEFF_FIXED):
    """
    Equation of a plane of form a*x + b*y + c*z + d = 0.

    X:  a 2D numpy array where X[:,0] = x and X[:,1] = y

    return: z
    """
    x = X[0]
    y = X[1]
    return (x * a + y * b + d) / -c


class Plane:
    """
    Class representing a plane in 3D.

    Mathematically the plane has the equation:
    ax + by + cz = d. The input paramaters are exactly these four coefficients.
    """

    def __init__(self, a: float, b: float, c: float, d: float) -> NoReturn:

        self._a = a
        self._b = b
        self._c = c
        self._d = d

    @classmethod
    def from_dataset(cls, x: np.array, y: np.array, z: np.array, p0: List[float] = [1.0, 1.0, 1.0]) -> Plane:
        """
        Create a Plane object from a dataset of x, y, z values

        This method uses scipy.optimize.curve_fit to fit the mathematical equation of a plane
        ax + by + cz = d to fit the dataset. The a, b, and d values are optimized while c is assumed to be
        1.0 or the value specified in PLANE_COEFF_FIXED.

        :param x: a numpy.array of the x-coordinates
        :param y: a numpy.array of the y-coordinates
        :param z: a numpy.array of the z-coordinates
        :param p0: initial values for the coefficients a, b, d
        :return:
        """
        X = np.array([x, y])
        ans, cov = curve_fit(equation_plane, X, z, p0=p0)
        return cls(ans[0], ans[1], PLANE_COEFF_FIXED, ans[-1])

    def plot_mayavi(self, minimums: List[float], maximums: List[float], color: RGB = (1., 0, 0),
                    **kwargs) -> NoReturn:
        """
        Plot the plane in a mayavi plot.

        :param minimums: the minimum values in x, y, z to be used for plotting
        :param maximums: the maximum values in x, y, z to be used for plotting
        :param color: the color the plane should be used for plotting
        :param kwargs: any kwargs passed to mayavi.mlab.surf
        :return: NoReturn
        """

        minx, miny, minz = minimums
        maxx, maxy, maxz = maximums

        X = np.mgrid[minx:maxx:10, miny: maxy: 10]
        z = equation_plane(X, self._a, self._b, self._d, self._c)
        mlab.surf(*X, -z, color=color, **kwargs)

    def intersect(self, other: Plane) -> Line:
        """
        Intersect with another Plane instance.

        Intersect two Planes in order to get the intersection Line.

        :param other: the Plane to intersect with.
        :return: A Line object
        """

        n1xn2 = np.cross(self.normal_vector, other.normal_vector)
        d = np.array([-self._d, -other._d, 0])

        A = [self.normal_vector, other.normal_vector, n1xn2]
        d = d.reshape(3, 1)

        p = np.linalg.solve(A, d)
        t = (n1xn2)

        return Line(p, t)

    def intersect_with_two_planes(self, other1: Plane, other2: Plane) -> Point:
        """
        Intersect the plane with two other Plane objects.

        The result is a Point.

        :params other1, other2: the two Planes to intersect with.
        :return: the point of intersection
        """

        A = np.array([self.coeffs[0], other1.coeffs[0], other2.coeffs[0]])
        B = np.array([self.coeffs[1], other1.coeffs[1], other2.coeffs[1]])
        C = np.array([self.coeffs[2], other1.coeffs[2], other2.coeffs[2]])
        D = np.array([self.coeffs[3], other1.coeffs[3], other2.coeffs[3]])

        det = np.linalg.det([A, B, C])

        x = np.linalg.det(np.array([D, B, C])) / -det
        y = np.linalg.det(np.array([A, D, C])) / -det
        z = np.linalg.det(np.array([A, B, D])) / -det
        res = Point(x, y, z)

        return res

    @property
    def coeffs(self) -> List[float]:
        """The four coefficients a, b, c, and d in the equation: ax + by + cz = d"""
        return [self._a, self._b, self._c, self._d]

    @property
    @lru_cache()
    def normal_vector(self) -> List[float]:
        """The Plane's normal vector (normalized)"""
        n = np.array(self.coeffs[:3])
        n = -n if n[-1] < 0 else n  # ensures that the last value (in z-direction) is always positive
        return n/np.linalg.norm(n)

    def z_from_xy(self, min_x: float, max_x: float, min_y:float , max_y: float, steps: int) -> np.array:
        X = np.mgrid[min_x: max_x: steps * 1j, min_y: max_y: steps * 1j]
        return *X, equation_plane(X, self._a, self._b, self._d, self._c)

    def __str__(self):
        return f'Plane: {self._a:.3g} * x + {self._b:.3g} * y + {self._c:.3g} * z = {self._d:.3g}'
