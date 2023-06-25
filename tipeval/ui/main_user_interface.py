"""
This module contains the main widget for the user interface.
"""
import os

import sys
import typing as T

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QMessageBox
from PyQt5 import uic, QtWidgets

import tipeval.ui.resources.ui_files
from tipeval.core.typing import FilePath
from tipeval.core.utils.data import get_resource_filename
from tipeval.ui.evaluation import Evaluation
from tipeval.ui.widgets import DataSelectionWidget
from tipeval.ui.widgets.fit_widget import FitWidget
from tipeval.ui.widgets.image_load_widget import ImageLoadWidget


BUTTON_LABELS = {Evaluation.State.LoadFile: 'Load file',
                 Evaluation.State.ChooseData: 'Choose data',
                 Evaluation.State.FitArea: 'Fit tip/area',
                 Evaluation.State.DetermineRadius: 'Determine tip radius',
                 Evaluation.State.SaveAreaFunction: 'Save area function'}


class MainUserInterface(QMainWindow):
    """
    The main user interface.
    """

    def __init__(self):

        super().__init__()

        from tipeval.config import configuration

        self.evaluation: T.Optional[Evaluation] = None
        self._evaluation_ongoing = False
        self._configuration = configuration

        # the widgets put into the StackedLayout on the right
        self._widgets: T.Dict[Evaluation.State, T.Optional[QWidget]]

        self._state_methods: T.Dict[Evaluation.State, T.Callable] = {Evaluation.State.LoadFile: self.load_file,
                                                                     Evaluation.State.ChooseData: self.choose_data,
                                                                     Evaluation.State.FitArea: self.fit_area}

        self._init_interface()
        self._connect_signals()

    def _init_widgets(self):
        self._widgets = {state: None for state in Evaluation.State}

    def _init_interface(self):
        with get_resource_filename(tipeval.ui.resources.ui_files, 'main_window.ui') as f:
            uic.loadUi(f, self)
        # with get_resource_filename(tipeval.ui.resources, 'Combinear.qss') as f:
        #     with open(f, 'r') as s:
        #         self.qss = s.read()
        #     self.setStyleSheet(self.qss)

        self.back_button.set_icon('arrow_left.png')
        self.forward_button.set_icon('arrow_right.png')

        self._init_widgets()
        self._add_buttons()

    def _add_buttons(self):
        for i, state in enumerate(Evaluation.State):
            button = self.side_bar.add_button(BUTTON_LABELS[state], state=state)
            button.is_clicked.connect(self._set_section)
        self.side_bar._layout.addStretch()

    def _connect_signals(self):
        self.new_action.triggered.connect(self.new_evaluation)
        self.forward_button.clicked.connect(self.next_section)
        self.back_button.clicked.connect(self.previous_section)

    def _alert_ongoing_evaluation(self):
        answer = QMessageBox.information(self, 'Evaluation is ongoing', f'Would you like to start a new '
                                                                        f'evaluation and close the current one?',
                                         QMessageBox.Yes | QMessageBox.No)
        return answer == QMessageBox.Yes

    def _add_widget(self, state: Evaluation.State, widget: QWidget):
        self._widgets[state] = widget
        self.analysis_container.addWidget(widget)
        self.analysis_container.setCurrentIndex(state.value)

    def _remove_widgets(self):
        # remove all the widgets from the container
        for i in range(self.analysis_container.count()-1, 0, -1):
            widget = self.analysis_container.widget(i)
            self.analysis_container.removeWidget(widget)
            if widget is not None:
                widget.deleteLater()
        self._init_widgets()

    def _set_buttons_inactive(self):
        for button in self.side_bar.buttons.values():
            button.setEnabled(False)
            button.set_active(False)
        self.forward_button.setEnabled(False)
        self.back_button.setEnabled(False)

    def _set_button_states(self, state: Evaluation.State):
        self.back_button.setEnabled(state != Evaluation.State.LoadFile)

        self.forward_button.setEnabled(state != Evaluation.State.SaveAreaFunction)

        for string, button in self.side_bar.buttons.items():
            if BUTTON_LABELS[state] == string:
                button.set_active(True)
            else:
                button.set_active(False)

    def _set_section(self, state: Evaluation.State):
        self.evaluation.state = state
        self._set_button_states(state)

        if self.evaluation.result(state) is None:
            self.forward_button.setEnabled(False)

        self._state_methods[state]()

    def _check_continue_to_fitting(self, info: T.Dict[str, T.Any]) -> bool:

        conditions = info['min'] != 0, info['max'] != np.inf, info['roi'][0] is not None

        if not all(conditions):
            text = ''.join([string for string, cond in zip(['minimum limit / ',
                                                            'maximum limit / ',
                                                            'region of interest'], conditions)
                            if not cond]).rstrip('/ ')
            reply = QMessageBox.information(self, 'Not all limits set', f'Do you want to continue without '
                                                                        f'setting the: \n\t' + text,
                                            QMessageBox.Yes, QMessageBox.No)
            return reply == QMessageBox.Yes
        return True

    def new_evaluation(self, *, file: T.Optional[FilePath] = None):
        if self._evaluation_ongoing:
            answer = self._alert_ongoing_evaluation()
            if not answer:
                return
            else:
                self.restart_evaluation()

        if file is None:
            file, _ = QFileDialog.getOpenFileName(self, options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if not file:
            return

        if not os.path.exists(file):
            QMessageBox.critical(self, 'File not found', f'Could not find file {file}!')
            return

        self._evaluation_ongoing = True

        widget = ImageLoadWidget(self, file=file,
                                 output_directory=
                                 os.path.join(os.path.split(file)[0],
                                              self._configuration.output_settings.default_output_folder))

        self._add_widget(Evaluation.State.LoadFile, widget)
        self.side_bar.buttons[BUTTON_LABELS[Evaluation.State.LoadFile]].set_active(True)
        widget.load_signal.connect(self.load_file)

    def restart_evaluation(self):

        self._remove_widgets()
        self._evaluation_ongoing = False
        self._set_buttons_inactive()
        self.evaluation = None

    def load_file(self):
        if self.evaluation is not None:
            self.analysis_container.setCurrentIndex(1)
            return

        widget = self._widgets[Evaluation.State.LoadFile]

        self.evaluation = Evaluation()
        file_information = widget.get_file_info()

        try:
            data, info = self.evaluation.load_file(file_information.copy())
        except ValueError:
            QMessageBox.critical(self, 'Cannot read file', f'Cannot read file {file_information["input_file"]}! '
                                                           f'Perhaps the file has a different delimiter or symbol '
                                                           f'for comments. Currently you have set:\n\n'
                                                           f'\t\t"{file_information["delimiter"]}"     and     '
                                                           f'"{file_information["comment_symbol"]}"    , respectively.')
        except FileNotFoundError:
            QMessageBox.critical(self, 'Invalid directory', f'The directory {file_information["output_directory"]} '
                                                            f'either does not exist or cannot be created. Please '
                                                            f'save to a different one. ')
        else:
            widget.set_data_and_image(data, info['new_unit'])
            self.forward_button.setEnabled(True)

    def choose_data(self):
        state = Evaluation.State.ChooseData

        if self._widgets[state] is None:
            widget = DataSelectionWidget()
            self._add_widget(state, widget)
            data, info = self.evaluation.result(Evaluation.State.LoadFile)
            unit = info['new_unit']
            widget.set_image_data(data, unit=unit)
            widget.corner_changed_signal.connect(self.corners_set)
            return

        self.analysis_container.setCurrentIndex(state.value)
        widget = self._widgets[state]
        corners = widget.image_widget.plot.corners
        self.corners_set(len(corners) == 3)  # todo implement correct number of corners

    def fit_area(self):
        state = Evaluation.State.FitArea

        if self._widgets[state] is None:
            widget = FitWidget()
            self._add_widget(state, widget)
            self.evaluation.fit_area()
            widget.set_evaluation(self.evaluation)
            widget.cross_section_widget.update_signal.connect(self.evaluation.fit_area)

            return

        self.analysis_container.setCurrentIndex(state.value)

    def corners_set(self, all_corners_set: bool):
        self.forward_button.setEnabled(all_corners_set)

    def next_section(self):
        if Evaluation.State(self.evaluation.state.value + 1) == Evaluation.State.FitArea:
            widget = self._widgets[Evaluation.State.ChooseData]
            info = widget.get_info()
            if not self._check_continue_to_fitting(info):
                return
            self.evaluation.choose_data(info)
        elif Evaluation.State(self.evaluation.state.value + 1) == Evaluation.State.DetermineRadius:
            ...

        self._set_section(Evaluation.State(self.evaluation.state.value+1))

    def previous_section(self):
        self._set_section(Evaluation.State(self.evaluation.state.value-1))


def user_interface():
    app = QApplication.instance()
    window = MainUserInterface()

    # load file
    # file = r'C:\1_Work\tipeval\examples\self_imaging.xyz'
    # window.new_evaluation(file=file)
    # window._widgets[Evaluation.State.LoadFile].load()
    # window.forward_button.clicked.emit()
    # widget = window._widgets[Evaluation.State.ChooseData]
    # widget.image_widget.plot.set_corners([(893, -538),
    #                                       (42, 872),
    #                                       (-666, -337)])
    # widget.min_depth_field.setText('20')
    # widget.max_depth_field.setText('140')
    # widget.image_widget.set_roi([(-672, -255),
    #                              (  54,  758),
    #                              ( 608, -357),
    #                              (-644, -385)])
    #
    # window.forward_button.clicked.emit()

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    user_interface()
