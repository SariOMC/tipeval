from functools import wraps
import typing as T

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QDoubleSpinBox, QSpinBox, QLabel, QVBoxLayout, QHBoxLayout, QWidget, \
    QAbstractSpinBox

from tipeval.ui.utils import block_signals


_BOX_FORMAT = {'FixedWidth': 150,
              'Maximum': 2147483647,
              'Alignment': Qt.AlignRight}


boxes_dictionary = T.Dict[str, QAbstractSpinBox]


def apply_formatting_to_box(box: QAbstractSpinBox) -> T.NoReturn:
    """
    Apply the standard formatting to a spin box.

    The standard formatting is saved in the _BOX_FORMAT dictionary.

    :param box: the SpinBox to format
    :return: NoReturn
    """
    for prop, value in _BOX_FORMAT.items():
        getattr(box, f'set{prop}')(value)


def get_values(widgets: boxes_dictionary) -> T.Tuple[T.Union[int, float], ...]:
    return tuple(getattr(box, 'value')() for box in widgets.values())


@block_signals
def _calculate_start(*, widgets: boxes_dictionary) -> T.NoReturn:
    start, stop, step_size, number_steps = get_values(widgets)
    widgets['Step size'].setValue((stop - start) / number_steps)


@block_signals
def _calculate_end(*, widgets: boxes_dictionary) -> T.NoReturn:
    start, stop, step_size, number_steps = get_values(widgets)
    widgets['Step size'].setValue((stop - start) / number_steps)


@block_signals
def _calculate_step(*, widgets: boxes_dictionary) -> T.NoReturn:
    start, stop, step_size, number_steps = get_values(widgets)
    widgets['Number steps'].setValue((stop - start) / step_size)


@block_signals
def _calculate_number_steps(*, widgets: boxes_dictionary) -> T.NoReturn:
    start, stop, step_size, number_steps = get_values(widgets)
    widgets['Step size'].setValue((stop - start) / number_steps)


class RangeBoxWidget(QGroupBox):

    LABELS = ['Start', 'Stop', 'Step size', 'Number steps']

    def __init__(self, parent: QWidget = None, start: int = 1, stop: int = 100, step_size: int = 5) -> None:
        super().__init__(parent)

        self._start = start
        self._stop = stop
        self._step_size = step_size

        self._boxes = {label: box() for label, box in zip(self.LABELS, [QSpinBox,
                                                                        QSpinBox,
                                                                        QDoubleSpinBox,
                                                                        QSpinBox])}

        layout = QVBoxLayout()
        self.setLayout(layout)
        self._init_widgets()

    def _init_widgets(self) -> T.NoReturn:

        self._boxes['Start'].setValue(self._start)
        self._boxes['Start'].setMinimum(1)
        self._boxes['Stop'].setMaximum(self._stop)
        self._boxes['Stop'].setValue(self._stop)

        self._boxes['Step size'].setValue(self._step_size)
        self._boxes['Number steps'].setValue((self._stop-self._start)/self._step_size)

        for label, box in zip(self.LABELS, self._boxes.values()):
            apply_formatting_to_box(box)
            box.valueChanged.connect(self._update_ranges)
            setattr(box, 'name', label)

            layout = QHBoxLayout()
            layout.addWidget(QLabel(label))
            layout.addStretch()
            layout.addWidget(box)
            self.layout().addLayout(layout)

    @QtCore.pyqtSlot()
    def _update_ranges(self):
        """When the value of a spin box update the values in the boxes accordingly."""

        _update_functions = {label: func for label, func in zip(self.LABELS, [_calculate_start,
                                                                              _calculate_end,
                                                                              _calculate_step,
                                                                              _calculate_number_steps])}
        _update_functions[self.sender().name](widgets=self._boxes)

    def get_ranges(self) -> T.Tuple[T.Union[int, float], ...]:
        """
        Return the spin box values.

        :return: A tuple containing the integer/float values displayed in the spin boxes.
        """
        return get_values(self._boxes)
