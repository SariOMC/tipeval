""" This module contains the class for displaying the fit of the tip using plotly."""

import typing as T

import numpy as np
import plotly
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWebEngineWidgets import QWebEngineView

from tipeval.core.typing import FilePath
from tipeval.core.utils.display import fit_plot_plotly


class FitTipWidget(QWebEngineView):
    """
    Widget for displaying the fit of the tip to the data.
    """

    def __init__(self, *args, **kwargs):
        """
        all args and kwargs are passed to QWebEngineView
        """

        super().__init__(*args, **kwargs)
        self.figure: T.Optional[plotly.graph_objs.Figure] = None

    def set_plot(self, tip: 'Tip', fit_data: np.array) -> T.NoReturn:
        """
        Set the plot in the reserved area for plotting.

        This plots the 3D fit of a tip to the data used for fitting
        with plotly.

        :param tip: the instance of the Tip.
        :param fit_data: the data used for fitting the Tip instance.
        :return: NoReturn
        """

        self.figure, html = fit_plot_plotly(tip, fit_data)
        self.setHtml(html)

    def save_figure(self, path: FilePath) -> T.NoReturn:
        """
        Save the displayed figure as a file.

        :param path: the path to save the file to.
        :return: NoReturn
        """

        dpi = 300/0.0254

        size = self.contentsRect()
        pixmap = QPixmap(size.width(), size.height())
        self.render(pixmap)
        img = QImage(pixmap)
        img.setDotsPerMeterX(dpi)
        img.setDotsPerMeterY(dpi)
        img.save(path, quality=100)
