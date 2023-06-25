"""
Module containing the class for the buttons on the left side of the
user interface.
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton

from tipeval.ui.evaluation import Evaluation


class SectionButton(QPushButton):
    """
    Class for the buttons on the left of the user interface.

    Just some basic common formatting to QPushButton is applied.
    """

    is_clicked = pyqtSignal(Evaluation.State)

    def __init__(self, *args, state=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.setMinimumSize(350, 50)
        self.setMinimumWidth(350)

        self.is_active = False

        self.state = state

        font = QFont()
        font.setPixelSize(20)
        self.setFont(font)
        self.setEnabled(False)
        self.setStyleSheet("background-color: rgba(150, 150, 150, 255)")

        self.clicked.connect(self.clicked_)

    def set_active(self, active: bool) -> None:

        if active:
            self.setEnabled(True)
            self.setStyleSheet("background-color: #b78620")
            self.is_active = True
        else:
            self.setStyleSheet("background-color: rgba(150, 150, 150, 255)")
            self.is_active = False

    def clicked_(self):
        self.is_clicked.emit(self.state)