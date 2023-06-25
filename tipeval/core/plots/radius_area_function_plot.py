"""
This module contains the class that lets you plot the ratio of the measured contact  area and ideal area
from which the tip radius can be determined.
"""

import typing as T

import numpy as np

from . import AreaFitPlot


class RadiusAreaFunctionPlot(AreaFitPlot):
    """
    Class for plotting the determination of the tip radius from the ratio of fitted area function to ideal area.
    """
    def __init__(self, contact_depths: np.ndarray, areas: np.ndarray, fit_coefficients: T.List[float],
                 ideal_fit_coefficients: T.List[float], hs: float, factor: float, unit: str = '-') -> None:
        """
        :param contact_depths: the contact depths where the areas were determined
        :param areas: the determined contact areas
        :param fit_coefficients: the coefficients of the polynomial fit
        :param ideal_fit_coefficients: the coefficients of the ideal area. Needs to be supplied if the area ratio
        is to be plotted
        :param hs: the resulting contact depth from which the tip radius can be determined
        :param factor: the area ratio at hs
        :param unit: the unit of the contact depth and areas (only one unit is allowed and squared for the area)
        """

        self._contact_depths = contact_depths
        self._areas = areas
        self.fit_coefficients = fit_coefficients
        self._ideal_fit_coefficients = ideal_fit_coefficients
        self._hs = hs
        self._factor = factor
        self._unit = unit

        super().__init__(self._contact_depths, self._areas, self.fit_coefficients, True, self._ideal_fit_coefficients,
                         unit=self._unit)
        self._plot_hs()

    def _plot_hs(self):
        """Plot the point of the area ratio from which the radius can be determined"""
        self._ax.set_xlim(self._ax.get_xlim())
        self._ax.set_ylim(self._ax.get_ylim())

        min_x = min(self._ax.get_xlim())
        min_y = min(self._ax.get_ylim())

        self._ax.plot([min_x, self._hs, self._hs], [self._factor, self._factor, min_y], 'bo--', markevery=[1], mfc='w',
                      mec='b', label=r'$A/A_i = $' + f'{round(self._factor, 2)}')
