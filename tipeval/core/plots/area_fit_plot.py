import typing as T

import matplotlib.pyplot as plt
import numpy as np

from tipeval.core.typing import FilePath
from tipeval.core.utils.nanoindentation import area_function_polynomial
from tipeval.core.plots.base_plot import BasePlot


class AreaFitPlot(BasePlot):
    """
    Class fot plotting the fit of the measured contact area
    """
    def __init__(self, contact_depths, areas: np.array, coefficients: T.List[float], plot_ratio: bool,
                 ideal_coefficients: T.List[float] = (), unit: str = '-') -> None:
        """
        :param contact_depths: the contact depths where the areas were determined
        :param areas: the determined contact areas
        :param coefficients: the coefficients of the polynomial fit
        :param plot_ratio: if True, the ratio of the measured/ideal area is plotted rather than the area
        :param ideal_coefficients: the coefficients of the ideal area. Needs to be supplied if the area ratio
        is to be plotted
        :param unit: the unit of the contact depth and areas (only one unit is allowed and squared for the area)
        """

        super().__init__()

        self._ax = self._fig.add_subplot()

        self._contact_depths = contact_depths
        self._areas = areas
        self._coefficients = coefficients
        self._ideal_coefficients = ideal_coefficients

        self._plot_ratio = plot_ratio
        self._unit = unit

        if self._plot_ratio and not ideal_coefficients:
            raise ValueError("If the ratio needs to be plotted you need to supply the coefficients of the polynomial "
                             "for the ideal area.")

        self._set_up_plot()
        self._plot_data()
        self._plot_fit()

    def _set_up_plot(self):
        x_label = f'contact depth ({self._unit})'
        y_label = f'area ratio (-)' if self._plot_ratio else f'contact area ({self._unit})²'

        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(y_label)

    def _plot_data(self):
        if self._plot_ratio:
            y = self._areas/area_function_polynomial(self._contact_depths, *self._ideal_coefficients)
        else:
            y = self._areas

        self._ax.plot(self._contact_depths, y, 'k*', label='measured')

    def _plot_fit(self):
        x_fit = np.linspace(min(self._contact_depths), max(self._contact_depths), 100)

        if self._plot_ratio:
            y_fit = (area_function_polynomial(x_fit, *self._coefficients)
                     / area_function_polynomial(x_fit, *self._ideal_coefficients))
        else:
            y_fit = area_function_polynomial(x_fit, *self._coefficients)

        self._ax.plot(x_fit, y_fit, 'r-', label='fit')

    def show(self, legend: bool = True) -> T.NoReturn:
        """
        Show the figure.

        :param legend: if True the legend is plotted
        :return: NoReturn
        """

        if legend:
            self._ax.legend()
        plt.tight_layout()
        plt.show()

    @property
    def figure(self):
        """The figure containing the plot."""
        return self._fig

    @property
    def ax(self):
        """The axis object containing the plot."""
        return self._ax

