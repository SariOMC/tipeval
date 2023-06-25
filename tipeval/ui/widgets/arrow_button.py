"""
This module contains the button used for the arrows in the bottom of the user interface
"""

import typing as T

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QPushButton

from tipeval.core.typing import FilePath
from tipeval.core.utils.data import get_resource_filename
import tipeval.ui.resources.icons


class ArrowButton(QPushButton):
    """
    This class is basically just a button with a png icon and no text.

    It is used in the user interface for the two arrows in the bottom.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEnabled(False)

    def set_icon(self, image: FilePath) -> T.NoReturn:
        """
        Set the icon to the button.

        :param image: the path to a png file.
        :return: NoReturn
        """

        self.setText('')
        self.setIconSize(QSize(100, 50))
        with get_resource_filename(tipeval.ui.resources.icons, image) as f:
            i = QIcon()
            i.addPixmap(QPixmap(str(f)))
            self.setIcon(i)
