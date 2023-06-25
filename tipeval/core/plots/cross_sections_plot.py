"""
Module containing the class for plotting the cross sections.
"""

from itertools import cycle
import typing as T

import numpy as np
import matplotlib.pyplot as plt

from tipeval.core.plots.base_plot import BasePlot


class CrossSectionsPlot(BasePlot):
    """
    Class for plotting the cross sections
    """
    def __init__(self, contact_depths: np.array, cross_sections: np.array, colors: T.List[T.Tuple[float, ...]],
                 unit: str = '-', linewidth: float = 1) -> None:
        """
        :param contact_depths: the contact depths corresponding to the cross sections
        :param cross_sections: the cross sections. Have to be numpy arrays
        :param colors:
        :param unit:
        :param linewidth:
        """
        super().__init__()

        self._ax = self._fig.add_subplot()

        self._contact_depths = contact_depths
        self._cross_sections = cross_sections
        self._colors = colors
        self._unit = unit
        self._linewidth = linewidth
        self._linestyles = cycle(('-', '--'))

        self._set_up_plot()
        self._plot_cross_sections()

    def _set_up_plot(self):
        self._ax.set_xlabel(f'x ({self._unit})')
        self._ax.set_ylabel(f'y ({self._unit})')

    def _plot_cross_sections(self):

        for distance, cs, color, linestyle in zip(self._contact_depths, self._cross_sections,
                                                  self._colors, self._linestyles):
            if isinstance(distance, float):
                distance = round(distance, 1)
            self._ax.plot(*cs, c=color, label=f'{distance}', linestyle=linestyle, linewidth=self._linewidth)

    def show(self, legend: bool = False) -> T.NoReturn:
        """
        Show the plot

        :param legend: if True a legend is plotted. This might look awkward in case there are many cross
        sections plotted.
        :return: NoReturn
        """

        if legend:
            self._ax.legend()
        self._ax.set_aspect('equal')
        self._ax.invert_yaxis()
        plt.show()
