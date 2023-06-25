"""
This module supplies a stand-alone user interface for selecting the data to be fitted from a 3D image data set.
It can also be implemented into an existing GUI application.
"""
import typing as T

import h5py
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QMessageBox, QPushButton, QDesktopWidget, QGroupBox, \
    QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QCloseEvent
from pyqtgraph.graphicsItems.ROI import Handle
import pyqtgraph.exporters as exporters

from tipeval.core.utils.data import get_resource_filename
from tipeval.core.typing import FilePath
from tipeval.core.plots import ImagePlot
import tipeval.ui.resources.ui_files


def run_data_selection(file: T.Optional[FilePath] = None) -> T.NoReturn:
    """
    Convenience function that allows to run the data_selection_widget as standalone GUI.

    :param file: a file to be loaded directly into the User Interface.
    :return: NoReturn
    """

    if file == None:
        file = ''

    app = QApplication([])
    main = DataSelectionWidget()

    # adding an exit button
    exit_button = QPushButton('Exit', main)
    exit_button.clicked.connect(main.close)
    main.corners_layout.addWidget(exit_button)

    main.add_standalone_buttons()

    # if a file has been supplied this is being loaded.
    if file:
        main.set_image_data_from_file(file)
    main.show()
    app.exec()


class DataSelectionWidget(QWidget):
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

        with get_resource_filename(tipeval.ui.resources.ui_files, 'data_selection_widget.ui') as f:
            uic.loadUi(f, self)

        # get the configuration from the config file
        from tipeval.config import configuration
        self._configuration = configuration

        self._set_size()
        self._connect_signals()
        self._data_file: T.Optional[FilePath] = None
        self._add_icons()

        self._has_cropped_data = False

    def _set_size(self) -> T.NoReturn:
        """Set size depending on screen geometry"""
        dimensions = QDesktopWidget().screenGeometry(-1)
        *_, screen_width, screen_height = dimensions.width(), dimensions.height()
        self.setMinimumSize(int(screen_width*0.5), int(screen_height * 0.5))

    def add_standalone_buttons(self):
        """
        Adds all the buttons necessary when the widget is run as a separate program.
        """

        box = QGroupBox('Image')
        self.bottom_layout.addWidget(box)

        self.load_image_button = QPushButton('Load image')
        self.save_image_button = QPushButton('Save image')

        self.load_image_button.clicked.connect(self.load_image)
        self.save_image_button.clicked.connect(self.save_image)

        layout = QHBoxLayout()
        box.setLayout(layout)
        layout.addWidget(self.load_image_button)
        layout.addWidget(self.save_image_button)

        self.export_corners_button = QPushButton('Save corners')
        self.export_corners_button.clicked.connect(self.save_corners)
        self.horizontalLayout_3.addWidget(self.export_corners_button)

        self.save_limits_button = QPushButton('Save limits')
        self.save_limits_button.clicked.connect(self.save_limits)
        self.exclude_region_box.layout().addWidget(self.save_limits_button)

        self.save_roi_button = QPushButton('Save ROI data')
        self.save_roi_button.clicked.connect(self.save_roi_data)
        self.roi_box.layout().addWidget(self.save_roi_button)

    def _connect_signals(self) -> T.NoReturn:
        """Connect the signals of the contained widgets"""

        # a dictionary with all the slots that need to be connected to the respective
        # button clicked signals
        button_slots = {self.level_button: self.apply_iso_value,
                        self.hide_iso_button: self.hide_iso_line,
                        self.show_roi_button: self.show_roi,
                        self.show_roi_image_button: self.show_roi_data,
                        self.remove_corners_button: self.image_widget.plot.clear,
                        self.hide_corners_button: self.show_corners,
                        self.show_angles_button: self.show_angles,
                        self.set_minimum_button: self.apply_minimum,
                        self.set_maximum_button: self.apply_maximum,
                        self.hide_lines_button: self.hide_lines}

        for button, function in button_slots.items():
            button.clicked.connect(function)

        # the other signals
        self.image_widget.level_changed_signal.connect(self._set_iso_value)
        self.iso_edit.returnPressed.connect(self.apply_iso_value)
        self.image_widget.plot.corners_changed_signal.connect(self.update_corners)
        self.angle_box.valueChanged.connect(self.show_angles)

    def _add_icons(self) -> T.NoReturn:
        """Add icons to the buttons which should contain icons

        A dictionary with available icons is saved in ICONS which can
        be found in .
        """

        button_icons = {self.set_maximum_button: 'arrow',
                        self.set_minimum_button: 'arrow',
                        self.hide_lines_button: 'eye open',
                        self.show_angles_button: 'eye closed',
                        self.hide_corners_button: 'eye open',
                        self.show_roi_button: 'eye closed',
                        self.hide_iso_button: 'eye open'}

        for button, icon in button_icons.items():
            self._apply_icon(button, icon)

    @staticmethod
    def _apply_icon(button: QPushButton, icon: str) -> T.NoReturn:
        """Apply an icon to a button"""

        from tipeval.ui.resources.icons import ICONS

        i = QIcon()
        with get_resource_filename('tipeval.ui.resources.icons', ICONS[icon]) as file:
            i.addPixmap(QPixmap(str(file)))
        button.setIcon(i)

    def _enable_iso_line(self, enabled: bool) -> T.NoReturn:
        """Enable the box containing the control for the iso line"""
        self.iso_line_box.setEnabled(enabled)

    def _enable_roi(self, enabled: bool) -> T.NoReturn:
        """Enable the box containing the control for the ROI"""
        self.roi_box.setEnabled(enabled)
        if hasattr(self, 'save_roi_button'):
            self.save_roi_button.setEnabled(False)
        self.show_roi_image_button.setEnabled(False)

    def _enable_exclude_region(self, enabled: bool) -> T.NoReturn:
        """Enable the box containing the control for the minimum
        and maximum iso lines"""
        self.exclude_region_box.setEnabled(enabled)

    def _enable_all(self, enabled: bool = True):
        """Enable/disable all boxes."""

        self._enable_roi(enabled)
        self._enable_iso_line(enabled)
        self._enable_exclude_region(enabled)

        if hasattr(self, 'save_image_button'):
            self.save_image_button.setEnabled(enabled)

    def _set_iso_value(self, value: int) -> T.NoReturn:
        """Set the iso value to the respective QLineEdit."""
        self.iso_edit.setText(str(value))

    def _apply_min_max(self, which: str) -> T.NoReturn:
        """Apply the value from the level QLineEdit to either the
        minimum or maximum line."""

        value = self.iso_edit.text()
        try:
            value = int(value)
        except ValueError:
            QMessageBox.critical(self, 'Invalid value', 'Please supply an integer number!')
            return

        if which == 'min':
            self.min_depth_field.setText(str(value))
            self.image_widget.set_min_iso_line(value)
        if which == 'max':
            self.max_depth_field.setText(str(value))
            self.image_widget.set_max_iso_line(value)

    def _get_roi(self) -> [T.Optional[np.array], T.List[Handle]]:
        """Get the information of the ROI if it is displayed."""

        roi_data = self.image_widget.get_roi_data()
        roi_coordinates = [] if roi_data is None else self.image_widget.roi.getHandles()
        return roi_data, roi_coordinates

    def load_image(self) -> T.NoReturn:   # todo implement loading all data (corners, limits, ...)
        """Load an image file.

        The image file that is chosen has to be an hdf file
        previously generated using ImageData"""

        file_dialog = QFileDialog(self)
        data_file, _ = file_dialog.getOpenFileName()
        if data_file:
            self.set_image_data_from_file(data_file)

    def set_image_data(self, data, unit):
        self.image_widget.set_image_data(data, unit=unit)
        self.image_widget.plot.vb.autoRange()

        self._enable_all()

    def set_image_data_from_file(self, file: FilePath) -> T.NoReturn:
        """
        Load the image data from the file.

        The loaded data is then displayed in the image

        :param file: has to be a file previously generated using ImageData
        :return: NoReturn
        """

        if file:
            try:
                with h5py.File(file, 'r') as f:
                    data = f[self._configuration.hdf_keys["converted_subgroup"]]['converted_data'][:]
                    unit = f[self._configuration.hdf_keys["converted_subgroup"]].attrs['unit']
            except OSError:
                QMessageBox.information(self, 'Cannot open file', 'Cannot open file. This does not seem to be an'
                                                                  ' hdf file.')
            except KeyError:
                QMessageBox.information(self, 'Cannot open file', 'Cannot open file. The file does not contain the '
                                                                  'required hdf keys. Possibly this file has not '
                                                                  'been generated using ImageData or has otherwise '
                                                                  'been corrupted. ')
            else:
                self._data_file = file
                self.set_image_data(data, unit)

    def save_image(self) -> T.NoReturn:
        """Save the plot scene as png"""
        file_dialog = QFileDialog(self)
        save_file, _ = file_dialog.getSaveFileName(self, 'Save image', filter="Image file (*.png)")

        if not save_file:
            return

        if not save_file.endswith('.png'):
            save_file += '.png'

        exporter = exporters.ImageExporter(self.image_widget.plot)
        exporter.export(save_file)

    def hide_iso_line(self) -> T.NoReturn:
        """
        Hide the is line displayed in the image.

        :return: NoReturn
        """
        self.image_widget.show_iso_line()
        if self.hide_iso_button.isChecked():
            self._apply_icon(self.hide_iso_button, 'eye open')
        else:
            self._apply_icon(self.hide_iso_button, 'eye closed')

    def apply_iso_value(self) -> T.NoReturn:
        """
        Apply the value in the QLineEdit in the Iso line Box to the iso line.

        :return: NoReturn
        """

        value = self.iso_edit.text()
        try:
            value = int(value)
        except ValueError:
            QMessageBox.critical(self, 'Invalid value', 'Please supply an integer number!')
        else:
            self.image_widget.update_iso_curve(value=value)
            self.hide_iso_button.setChecked(True)

        self._apply_icon(self.hide_iso_button, 'eye open')

    def show_roi(self) -> T.NoReturn:
        """
        Display a draggable ROI allowing to choose a data subset from the image.

        :return: NoReturn
        """
        if self.show_roi_button.isChecked():
            self._apply_icon(self.show_roi_button, 'eye open')
        else:
            self._apply_icon(self.show_roi_button, 'eye closed')

        self.image_widget.show_roi()
        roi_is_shown = self.image_widget.roi.isVisible()
        if hasattr(self, 'save_roi_button'):
            self.save_roi_button.setEnabled(roi_is_shown)
        self.show_roi_image_button.setEnabled(roi_is_shown)

    def show_roi_data(self) -> T.NoReturn:
        """
        Open a new window showing the data enclosed by the ROI.

        :return: NoReturn
        """

        data = self.image_widget.get_roi_data()

        # remove the part of the ROI that is outside the image by removing the
        # columns and rows that are 0 only
        mask1 = np.all(data[-1] == 0, axis=1)
        data = data[:, ~mask1]
        mask2 = np.all(data[-1] == 0, axis=0)
        data = data[:, :, ~mask2]
        plot = ImagePlot(*data)
        plot.show()

    def show_corners(self) -> T.NoReturn:
        """
        Display the corner plots that are set by double clicking.

        :return: NoReturn
        """

        if self.hide_corners_button.isChecked():
            self._apply_icon(self.hide_corners_button, 'eye open')
            self.image_widget.plot.show_corners()
        else:
            self._apply_icon(self.hide_corners_button, 'eye closed')

            self.image_widget.plot.hide_corners()

    def show_angles(self) -> T.NoReturn:
        """
        Show/hide the omit region around the edges.

        Show/hide the lines around the edges that indicate how much
        of the region around the edges is omitted for fitting the tips.

        :return: NoReturn
        """

        self.image_widget.plot.clear_omit_ranges()
        if self.show_angles_button.isChecked():
            self._apply_icon(self.show_angles_button, 'eye open')
            self.image_widget.plot.draw_omit_ranges(int(self.angle_box.value()))
        else:
            self._apply_icon(self.show_angles_button, 'eye closed')

    def hide_lines(self) -> T.NoReturn:
        """
        Show/hide the lines indicating the maximum and minimum depth used for further processing.

        :return: NoReturn
        """

        if self.hide_lines_button.isChecked():
            self._apply_icon(self.hide_lines_button, 'eye open')
            self.image_widget.show_max_iso_line()
            self.image_widget.show_min_iso_line()
        else:
            self.image_widget.hide_max_iso_line()
            self.image_widget.hide_min_iso_line()
            self._apply_icon(self.hide_lines_button, 'eye closed')

    def apply_minimum(self) -> T.NoReturn:
        """
        Update the current iso line value to the minimum iso line.

        :return: NoReturn
        """

        self._apply_min_max('min')

    def apply_maximum(self) -> T.NoReturn:
        """
        Update the current iso line value to the maximum iso line.

        :return: NoReturn
        """

        self._apply_min_max('max')

    def update_corners(self) -> T.NoReturn:
        """
        Update the corners when double clicking into the image.

        This method takes care of how many corners need to be plotted and which
        of the buttons are allowed to be pressed. It is only possible to save the
        corners if the required amount of corners is reached.

        :return: NoReturn
        """

        self.corner_list.clear()
        corners = self.image_widget.plot.corners
        angles = self.image_widget.plot.angles

        # to enable removing the corners there must be at least one entry
        self.remove_corners_button.setEnabled(bool(corners))
        self.hide_corners_button.setEnabled(bool(corners))
        self._apply_icon(self.hide_corners_button, 'eye open')

        # exporting the buttons is only allowed when all corners were set
        if hasattr(self, 'export_corners_button'):
            self.export_corners_button.setEnabled(self.image_widget.plot.all_corners_set)

        for i, (corner, angle) in enumerate(zip(corners, angles)):
            self.corner_list.addItem(f'Corner {i+1}: x={int(corner[0])}; y={int(corner[1])}; Angle = {angle:.1f}°')

        if self.show_angles_button.isChecked():
            self.show_angles()

        self.hide_corners_button.setChecked(True)
        self.corner_changed_signal.emit(self.image_widget.plot.all_corners_set)

    def get_info(self):
        info = dict(
            roi=self._get_roi(),
            min=float(self.min_depth_field.text() or 0.),
            max=float(self.max_depth_field.text() or np.inf),
            angle_around_edges=float(self.angle_box.value()),
            corners=self.image_widget.plot.corners,
            corner_angles=self.image_widget.plot.angles
        )
        return info

    def save_roi_data(self, silent: bool = False) -> T.NoReturn:
        """
        Save the data contained in the ROI.

        Saves the data contained in the ROI to the loaded data file.

        :param silent: If True, then no Message is displayed after saving the data. This
        is mainly supposed to be used together with 'save all'
        :return: NoReturn
        """

        data = self.image_widget.get_roi_data()

        if data is None:
            reply = QMessageBox.question(self, 'No ROI data', 'Data has not been cropped using a region of interest. '
                                                              'Do you want to continue without cropping data?',
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.No:  # In case we do not want to continue exiting we raise an InterruptedError
                raise InterruptedError()
            else:
                return

        with h5py.File(self._data_file, 'a') as f:
            try:
                group = f[self._configuration.hdf_keys["crop_data_subgroup"]]
            except KeyError:
                group = f.create_group(self._configuration.hdf_keys["crop_data_subgroup"])
            else:
                try:
                    del group['roi_data']
                except KeyError:
                    pass

            group['roi_data'] = data
            group.attrs['unit'] = self.image_widget.unit

        self._has_cropped_data = True

        if not silent:
            QMessageBox.information(self, 'Data saved', f'ROI data has been saved to {self._data_file}')

    def save_limits(self, silent: bool = False) -> T.NoReturn:
        """
        Save the limits set using the minimum and maximum lines.

        Saves the limits set using the minimum and maximum lines as well
        as the angular range omitted around the edges.

        :param silent: If True, then no Message is displayed after saving the data. This
        is mainly supposed to be used together with 'save all'
        :return: NoReturn
        """

        with h5py.File(self._data_file, 'a') as f:
            try:
                f.create_group(self._configuration.hdf_keys.crop_data_subgroup)
            except ValueError:      # this error occurs when 'limits' already exists
                pass
            group = f[self._configuration.hdf_keys.crop_data_subgroup]

            min_ = self.min_depth_field.text() or 0.
            max_ = self.max_depth_field.text() or np.inf

            try:
                max_ = float(max_)
                min_ = float(min_)
            except ValueError:
                QMessageBox.information(self, 'Wrong input', 'Please only supply numbers for the limits.')
                return

            if max_ == min_:
                QMessageBox.information(self, 'Wrong input', 'Minimum and maximum should not be the same because in '
                                                             'this case no data will be available for fitting.')
                return

            if max_ < min_:
                QMessageBox.information(self, 'Minimum larger than maximum',
                                        'WARNING: Minimum is larger than maximum. Values will be exchanged. ')
                min_, max_ = max_, min_
                silent = True

            group.attrs['upper_limit'] = max_
            group.attrs['lower_limit'] = min_
            group.attrs['omit_angle'] = float(self.angle_box.value())

        if not silent:
            QMessageBox.information(self, 'Limits saved', f'The limits were successfully saved to {self._data_file}.')

    def save_corners(self, silent: bool = False) -> T.NoReturn:
        """
        Save the corners set by double clicking.

        Saves the corners set by double clicking to the loaded data file.

        :param silent: If True, then no Message is displayed after saving the data. This
        is mainly supposed to be used together with 'save all'
        :return: NoReturn
        """

        with h5py.File(self._data_file, 'a') as f:
            try:
                subgroup = f.create_group(self._configuration.hdf_keys["corner_subgroup"])
            except ValueError:
                subgroup = f[self._configuration.hdf_keys["converted_subgroup"]]

            try:
                del subgroup['coordinates']
            except KeyError:
                pass

            subgroup['coordinates'] = np.array(self.image_widget.plot.corners)

        if not silent:
            QMessageBox.information(self, 'Corners saved', f'The corners were successfully saved to {self._data_file}.')

    def save_all(self) -> T.NoReturn:
        """
        Save all to the data file.

        All includes:
        - the corners
        - the limits
        - the ROI data
        If no ROI data has been previously chosen, a Message is raised asking for confirmation.

        :return: NoReturn
        """
        if not self._has_cropped_data:
            self.save_roi_data(silent=True)

        if not self.image_widget.plot.all_corners_set:
            reply = QMessageBox.question(self, 'Not all corners set', 'You did not yet define all the corners in the '
                                                                      'image. Would you like to proceed anyway?',
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        else:
            self.save_corners(silent=True)

        self.save_limits(silent=True)

    def closeEvent(self, event: QCloseEvent) -> T.NoReturn:
        """
        Exit the application.

        This is only intended to be used together with the standalone application.
        started by the run_data_selection function.

        :return: NoReturn
        """

        if self._data_file is None:
            reply = QMessageBox.question(self, 'Exit', 'Do you want to exit the application?',
                                         QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.close()
        else:
            reply = QMessageBox.question(self, 'Exit', 'Do you want to save all modifications?',
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                try:
                    self.save_all()
                except InterruptedError:
                    event.ignore()
                    return
            event.accept()
