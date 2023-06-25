"""
Module containing the left side of the user interface which contains
the different buttons indicating the current progress in the evaluation.
"""
import typing as T

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from tipeval.ui.widgets.section_button import SectionButton
from tipeval.ui.evaluation import Evaluation


class SideBarWidget(QWidget):
    """
    Class for the Widget containing the buttons on the left side.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.buttons: T.Dict[str, SectionButton] = {}

    def add_button(self, string: str, state: Evaluation.State) -> SectionButton:

        button = SectionButton(string, state=state)
        button.setFixedWidth(250)
        self._layout.addWidget(button)
        self.buttons[string] = button

        return button
