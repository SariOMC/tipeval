"""
This module contains the widget for fitting the area function.
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel


class AreaFunctionFitWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel('c\'ya'))