""" This module contains the widget that contains the fit tip and associated functionality"""

import typing as T

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog

from tipeval.core.utils.data import get_resource_filename
import tipeval.ui.resources.ui_files
from tipeval.ui.evaluation import Evaluation


class FitWidget(QWidget):
    """
    Widget for choosing the data for further processing from a tips image.

    This widget contains the graphs and functionality necessary for evaluating a
    tips image. These include:
    - an image of the tips
    - a histogram
    - boxes for controlling the :
        - file containing the image data
        - iso lines indicating the maximum and minimum depth values
          to be used for further analysis
        - a region of interest (ROI) for cropping the data
        - corner plots indicating the edges of the tips

    This widget can either be implemented into an existing PyQt application or
    run as a standalone GUI using the function run_data_selection() also saved in
    this module.
    """

    corner_changed_signal = pyqtSignal(bool)

    def __init__(self, parent: T.Optional[QWidget] = None) -> T.NoReturn:
        """
        :param parent: the Qt parent widget
        """
        super().__init__(parent)

        with get_resource_filename(tipeval.ui.resources.ui_files, 'fit_widget.ui') as f:
            uic.loadUi(f, self)

        self.save_button.clicked.connect(self.save_image)

    def set_evaluation(self, evaluation: Evaluation) -> T.NoReturn:
        """
        Set the plot in the reserved area for plotting.

        This plots the 3D fit of a tip to the data used for fitting
        with plotly.

        :param evaluation: the Evaluation object used.
        :return: NoReturn
        """

        self.fit_tip_widget.set_plot(evaluation.tip, evaluation.fit_data)
        self.info_box.set_tip(evaluation.tip)
        self.cross_section_widget.set_evaluation(evaluation)


    def save_image(self) -> T.NoReturn:
        """
        Save the displayed image to a file.

        :return: NoReturn
        """

        file, _ = QFileDialog.getSaveFileName(self, options=QFileDialog.DontUseNativeDialog)

        if file:
            self.fit_tip_widget.save_figure(file)
