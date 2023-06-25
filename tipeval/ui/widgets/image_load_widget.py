""""""
from datetime import datetime
import os
from pathlib import Path
import typing as T

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog, QStackedLayout, QVBoxLayout

import tipeval.ui.resources.ui_files
from tipeval import ImagePlot, SobelPlot
from tipeval.core.typing import FilePath
from tipeval.core.utils.data import get_resource_filename


class ImageWidgetToolbar(QWidget):

    def __init__(self, parent=None, figure: plt.Figure = None, enabled: bool = True):
        super().__init__(parent=parent)

        self.image = ImageWidget(figure)
        self.toolbar = NavigationToolbar2QT(self.image.fig.canvas, parent)

        self._layout = QVBoxLayout()
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.image)
        self.setLayout(self._layout)
        self.setEnabled(enabled)


class ImageWidget(FigureCanvasQTAgg):

    def __init__(self, figure: plt.Figure = None):

        if figure is None:
            figure = plt.figure()

        self.fig = figure
        self.fig.tight_layout()
        super().__init__(self.fig)


class ImageLoadWidget(QWidget):
    """
    The widget used for loading the recorded image file and displaying it.
    """

    load_signal = pyqtSignal()

    def __init__(self, parent: QWidget = None, file: T.Optional[FilePath] = None,
                 output_directory: T.Optional[FilePath] = None) -> None:
        super().__init__(parent=parent)

        with get_resource_filename(tipeval.ui.resources.ui_files, 'image_file_widget.ui') as f:
            uic.loadUi(f, self)

        self.file: T.Optional[FilePath]
        self.output_directory: T.Optional[FilePath]
        self.image: T.Optional[ImagePlot] = None
        self.sobel_image: T.Optional[SobelPlot] = None
        self.image_data: T.Optional[np.array] = None
        self.unit = ''
        self.date: T.Optional[datetime] = None

        self._set_file(file)

        self._init_drop_lists()

        self._layout = QStackedLayout()
        self.right_widget.setLayout(self._layout)

        self._init_figure_canvas()
        self._connect_signals()

        self.set_output_directory(output_directory)
        date = datetime.fromtimestamp(Path(file).stat().st_mtime)
        self.set_recording_date(date)

    def _set_file(self, file: T.Optional[FilePath]):
        self.file = file
        if self.file is not None:
            self.file_field.setText(os.path.split(self.file)[-1])

    def _connect_signals(self):
        self.output_folder_button.clicked.connect(self._change_output_folder)
        self.load_button.clicked.connect(self.load)
        self.sobel_button.clicked.connect(self.switch_images)

    def _init_figure_canvas(self):

        self.image = ImageWidgetToolbar(enabled=False)
        self._layout.addWidget(self.image)
        self._layout.setCurrentIndex(0)

    def _init_drop_lists(self):
        """Initialize the possible drop-down values from the configuration."""
        from tipeval.core.utils.data import TO_UNIT
        from tipeval.config import configuration

        for unit in TO_UNIT.keys():
            self.base_unit_list.addItem(unit)

        for unit in list(TO_UNIT.keys())[::-1]:
            self.convert_unit_list.addItem(unit)
            self.convert_unit_list.setCurrentIndex(self.convert_unit_list.findText('nm'))

        for geometry in configuration.available_tips:
            self.type_list.addItem(geometry)

        for method in configuration.available_methods:
            self.method_list.addItem(method)

        for delimiter in configuration.delimiters:
            self.delimiter_list.addItem(delimiter)

        for comment_symbol in configuration.comment_symbols:
            self.comment_symbol_list.addItem(comment_symbol)

    def _change_output_folder(self):
        """Change the folder where the data is saved."""
        dialog = QFileDialog(self)
        output_directory = dialog.getExistingDirectory(self, options=(QFileDialog.DontUseNativeDialog |
                                                                      QFileDialog.ShowDirsOnly))
        if output_directory:
            self.set_output_directory(output_directory)

    def set_output_directory(self, output_directory: FilePath) -> T.NoReturn:
        """
        Set the output directory.

        :param output_directory:
        """
        if output_directory is not None:
            self.output_field.setText(os.path.abspath(output_directory))

    def set_recording_date(self, date: datetime) -> T.NoReturn:
        """
        Set the recording date of the QDateEdit

        :param date: the new date to be set
        """
        self.date_edit.setDateTime(date)
        self.date = date

    def load(self):

        self.load_signal.emit()

    def get_file_info(self) -> T.Dict[str, T.Any]:

        file_info = dict(input_file=self.file,
                         output_directory=self.output_field.text(),
                         imaging_method=self.method_list.currentText(),
                         date=datetime.combine(self.date_edit.date().toPyDate(), datetime.min.time()),
                         base_unit=self.base_unit_list.currentText(),
                         new_unit=self.convert_unit_list.currentText(),
                         tip_type=self.type_list.currentText(),
                         tip_name=self.name_field.text() or 'no_name',
                         delimiter=self.delimiter_list.currentText(),
                         comment_symbol=self.comment_symbol_list.currentText())
        return file_info

    def set_data_and_image(self, data, unit: str = '-'):

        self.image_data = data
        self.unit = unit
        self.load_button.setEnabled(False)
        self.sobel_button.setEnabled(True)
        self.output_folder_button.setEnabled(False)
        self.show_image()

    def show_image(self):
        if self.image is not None:
            self.image = ImagePlot(*self.image_data, unit=self.unit)
            widget = ImageWidgetToolbar(self, self.image.figure)
            self._layout.addWidget(widget)
        self._layout.setCurrentIndex(1)

    def show_sobel_image(self):
        """
        Display the sobel filtered image.
        """
        if self.sobel_image is None:
            self.sobel_image = SobelPlot(*self.image_data, unit=self.unit)
            widget = ImageWidgetToolbar(self, self.sobel_image.figure)
            self._layout.addWidget(widget)
        self._layout.setCurrentIndex(2)

    def switch_images(self):
        """
        Switch between the two possible images.
        """
        if self._layout.currentIndex() == 2:
            self.show_image()
            self.sobel_button.setText('Show Sobel-filtered image')
        else:
            self.show_sobel_image()
            self.sobel_button.setText('Show regular image')
