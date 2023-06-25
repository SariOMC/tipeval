import typing as T

import numpy as np


def equation_sphere(xy_coorinates: T.Tuple[np.array, np.array], x0: float, y0: float, z0: float, radius: float) -> np.array:
    """
    The equation of the surface of a sphere with radius and center at x0, y0, z0.

    The x and y coordinates have to be supplied as a tuple of coordinates having
    length 2 such that x, y = xy_coorinates.
    :return: the z-coordinates of the surface of the sphere
    """
    x, y = xy_coorinates
    return -np.sqrt(radius**2-(x-x0)**2 - (y-y0)**2) + z0
