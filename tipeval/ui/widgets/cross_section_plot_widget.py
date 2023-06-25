import typing as T

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class Plot(FigureCanvasQTAgg):

    def __init__(self, *args, **kwargs) -> T.NoReturn:
        self.figure = plt.figure()
        super().__init__(self.figure, *args, **kwargs)

    def set_data(self):
        ...


class CrossSectionPlotWidget(QWidget):

    def __init__(self, *args, **kwargs) -> T.NoReturn:
        super().__init__(*args, **kwargs)

        self.plot = Plot()
        self.toolbar = NavigationToolbar2QT(self.plot.figure.canvas, self)

    def _init_widgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.plot)

    def set_data(self):
        self.plot.set_data()
