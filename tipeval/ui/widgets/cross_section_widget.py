"""
This module contains the widget for choosing the depths for calculating the cross-sections.
"""
import typing as T

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QMessageBox

from tipeval.core.utils.data import get_resource_filename, calculate_cross_sections
from tipeval.ui.evaluation import Evaluation


class CrossSectionWidget(QWidget):

    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import tipeval.ui.resources.ui_files

        with get_resource_filename(tipeval.ui.resources.ui_files, 'cross_section_widget.ui') as f:
            uic.loadUi(f, self)

        self.update_button.clicked.connect(self.update_plot)

        self.evaluation: T.Optional[Evaluation] = None

    def check_values(self, start: int, stop: int, step_size: float, number_steps: int):
        if start >= stop or stop <= start:
            QMessageBox.critical(self, 'Wrong input', 'Please ensure that the start value is '
                                                      'smaller than the stop value and vice versa.')
            raise ValueError

    def set_evaluation(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def _get_ranges(self):
        values1 = self.range_box_1.get_ranges()
        values2 = ()

        depth_values = np.array([], dtype=float)

        if self.range_box_2.isChecked():
            values2 = self.range_box_2.get_ranges()

        for values in (values1, values2):
            if not values:
                continue
            try:
                self.check_values(*values)
            except ValueError:
                return
            start, stop, _, number_steps = values
            depth_values = np.concatenate((depth_values, np.linspace(start, stop, number_steps)))

        return np.unique(depth_values)

    def update_plot(self):
        depth_values = self._get_ranges()
        self.update_signal.emit(depth_values)

