import typing as T
import warnings

import numpy as np
from scipy.optimize import newton

from tipeval.core.typing import Depth


if T.TYPE_CHECKING:
    from tipeval.core.tips.three_sided_pyramids import ThreeSidedPyramidTip


# the equivalent cone half angles
OPENING_ANGLE_BERKOVICH = 65.27
IDEAL_CONE_ANGLE_BERKOVICH = 70.25

OPENING_ANGLE_CUBE_CORNER = 35.264
IDEAL_CONE_ANGLE_CUBE_CORNER = 42.0

STANDARD_COMPLIANCE = 0.210


# the different types of tips available in a hysitron area function
TIP_TYPES_HYSITRON = {
    'Berkovich': 0,
    'Cono-spherical': 1,
    'Cube-corner': 2,
    'Flat-punch': 3,
    'Knoop': 4,
    'Vickers': 5,
    'Other': 99
}

TIP_TYPES_UMIS = {
    'Berkovich': 'BE'
}

IDEAL_ANGLES = {
    'Berkovich': (OPENING_ANGLE_BERKOVICH, IDEAL_CONE_ANGLE_BERKOVICH),
    'Cube-corner': (OPENING_ANGLE_CUBE_CORNER, IDEAL_CONE_ANGLE_CUBE_CORNER)
}


def first_coefficient_polynomial(angle: float) -> float:
    """
    The first coefficient of the area function polynomial.

    This function calculates, for a given opening angle (or equivalent cone angle),
    the first coefficient C0 of the area function polynomial:

    A = C0*h**2 + C1*h + C2*h**(h/2) + C3*h**(h/4) + ...

    :param angle: the opening angle of the cone or equivalent cone angle
    :return: the first coefficient C0
    """
    return area_cone(1, angle)


def area_cone(depth: Depth, tip_angle: float) -> T.Union[np.array, float]:
    """
    Calculate the depth-dependent cross-section of a cone with opening angle *tip_angle*.

    :param depth: the depth(s) for which to calculate the cross-section.
    :param tip_angle: the angle of the cone in degrees
    :return: the cross section(s)
    """
    return 3*np.sqrt(3)*depth**2*np.tan(np.deg2rad(tip_angle))**2


def area_berkovich(depth: Depth) -> T.Union[np.array, float]:
    """
    Calculate the depth dependent cross-section of a Berkovich tip.

    :param depth: the depth(s) for which to calculate the cross-section.
    :return: the cross section(s)
    """
    return area_cone(depth, OPENING_ANGLE_BERKOVICH)


def area_cube_corner(depth: Depth) -> T.Union[np.array, float]:
    """
    Calculate the depth dependent cross-section of a cube corner tip.

    :param depth: the depth(s) for which to calculate the cross-section.
    :return: the cross section(s)
    """
    return area_cone(depth, OPENING_ANGLE_CUBE_CORNER)


def area_function_polynomial(depth: Depth, *coefficients: T.List[float]) -> np.array:
    """
    Calculate the area for an area function polynomial of the form:

    A = C0*h**2 + C1*h + C2*h**(h/2) + C3*h**(h/4) + ...

    This is a well known equation suggested in the original Oliver and Pharr paper.

    :param depth: the depth(s) for which to calculate the cross-section.
    :param coefficients: a list of the coefficients
    :return: the calculated are
    """
    area = 0
    for i, coeff in enumerate(coefficients):
        area += coeff * np.power(depth, 2 / np.power(2, i))
    return area


def area_function_polynomial_fixed(depth: Depth, *coefficients, coefficient0: float = 0.0) -> np.array:
    """
    Calculate the area for an area function polynomial of the form:

    A = C0*h**2 + C1*h + C2*h**(h/2) + C3*h**(h/4) + ...

    This is a well known equation suggested in the original Oliver and Pharr paper.
    Here C0 can be supplied and is fixed (mostly useful for fitting).

    :param depth: the depth(s) for which to calculate the cross-section.
    :param coefficients: a list of the coefficients
    :param coefficient0: the first coefficient C0
    :return: the calculated are
    """
    area = coefficient0 * depth ** 2
    for i, coeff in enumerate(coefficients):
        area += coeff * np.power(depth, 2 / np.power(2, (i+1)))
    return area


def find_zero(function: T.Callable, x0: float = 5) -> float:
    """
    Find the root for a function.

    This function uses the Newton method from *scipy.optimize*. In addition to the
    initial guess also its half and doubled values are tried.

    :param function: the function to find the root for
    :param x0: an initial guess
    :return:
    """

    max_iterations = 1000
    tolerance = 1E-5

    try:
        x0 = x0
        hs = newton(function, x0=x0, maxiter=max_iterations, tol=tolerance)
    except RuntimeError:
        try:
            hs = newton(function, x0=x0/2, maxiter=max_iterations, tol=tolerance)
        except RuntimeError:
            try:
                x0 = x0/4
                hs = newton(function, x0=2*x0, maxiter=max_iterations, tol=tolerance)
            except RuntimeError:
                raise RuntimeError(f'Finding root of the function is not possible with starting values {x0}, '
                                   f'{x0}/2 and 2*{x0}!')
    return hs


def contact_radius_sphere(depth: np.ndarray, radius: float) -> np.ndarray:
    """
    The contact radius of a sphere.

    :param depth: the contact depth.
    :param radius: the radius of the sphere.
    :return: the contact radius calculated for the given depth and radius
    """
    return np.sqrt(2 * radius * depth - depth ** 2)


def contact_radius_cone(depth: np.ndarray, angle: float, intercept: float) -> np.ndarray:
    """
    The contact radius of a cone.

    :param depth: the contact depth
    :param angle: the half angle of the cone in degrees
    :param intercept: a constant offset in case that the apex of the cone is not at 0, 0
    :return: the contact radius calculated for the given depth and radius
    """
    return depth * np.tan(np.deg2rad(angle)) + intercept


def opening2cone_angle_three_sided_pyramid(phi: float) -> float:
    """
    Convert the opening angle of a three sided pyramid to the equivalent cone angle.

    :param phi: the opening angle of the pyramid
    :return: the converted angle
    """
    return np.rad2deg(np.arctan(np.sqrt(3*3**0.5/np.pi*np.tan(np.deg2rad(phi))**2)))


def area_function_hysitron(tip: 'ThreeSidedPyramidTip') -> str:
    """Generate the text to be saved in a hysitron area function

    :param coefficients: the coefficients
    :param error_coefficients:
    :param type:
    :param tip_name:
    :param tip_radius:
    :param tip_angle:
    :return: the text saved in the area function file
    """
    type = TIP_TYPES_HYSITRON.get(tip.type, TIP_TYPES_HYSITRON['Other'])

    if type == 99:
        warnings.warn(f"Could not found {tip.type}. The tip type is therefore set to 'Other'.")

    text = f"""Hysitron Area Function Coefficients, File Version: 4.00
C0
{tip.fit_coefficients[0]}
C1
{tip.fit_coefficients[1]}
C2
{tip.fit_coefficients[2]}
C3
{tip.fit_coefficients[3]}
C4
{tip.fit_coefficients[4]}
C5
{tip.fit_coefficients[5]}
B
0.0
C0 Uncertainty
{tip.error_fit_coefficients[0]}
C1 Uncertainty
{tip.error_fit_coefficients[1]}
C2 Uncertainty
{tip.error_fit_coefficients[2]}
C3 Uncertainty
{tip.error_fit_coefficients[3]}
C4 Uncertainty
{tip.error_fit_coefficients[4]}
C5 Uncertainty
{tip.error_fit_coefficients[5]}
B Uncertainty
0.0
Tip Poisson's Ratio
0.07
Tip Modulus (N/mm^2)
1140.0
Tip Types: Berkovich=0 Cono-spherical=1, Cube-Corner=2, Flat-Punch=3, Knoop=4, Vickers=5, Other=99
Tip Type
{tip.type}
Max Events Between Cal
100
Events Since Last Cal
0
Total Events
0
Tip Serial Number
{tip.name}
Tip Radius (nm)
{tip.radius_blunting_depth}
Tip Total Angle
{2 * tip.equivalent_cone_angle}
Tip Usuable Depth (um)
13.0
"""
    return text


def area_function_umis(tip: 'ThreeSidedPyramidTip', compliance: T.Optional[str] = None, method: str = '') -> str:
    """Generate the text to be saved in a UMIS area function

    :param tip: the instance of the tip for which to save the area function
    :param compliance: the value of the compliance of the system. If not supplied then the value of STANDARD_COMPLIANCE
                       will be used.
    :param method: the method used for recording the data
    :return: the text saved in the area function file
    """

    method = method or 'optical calibration'
    if compliance is None:
        compliance = STANDARD_COMPLIANCE

    file_ending = '.afn'
    text = ''

    header = 'AF;{};{:.1f};{:.3f};0.035315;0.062;,hc,A/Ai(Raw),A/Ai(Fitted),dPdh/E/b,Tip radius\n'.format(
        TIP_TYPES_UMIS[tip.type], tip.tip_angle, compliance)

    text += header
    for i, (depth, area) in enumerate(zip(tip.contact_depths/1000, tip.area_ratios)):
        if i == 0:
            continue
        text += f'{tip.type.replace(" ", "_")}_{method.replace(" ", "_")}_{i},'
        text += f'{depth:.8f},{area:.6f},{area:.6f},0.50000000,{np.sqrt(area/np.pi):.5f}\n'

    return text
