"""
This file contains the widgets comprising the main graphics display
used further in the data_selection_widget
"""

from typing import Optional, List, NoReturn, Tuple

import numpy as np
from PyQt5 import QtCore
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget
from pyqtgraph.Point import Point

from tipeval.core.utils.data import angle2D


# Some basic configuration of the pyqtgraph plots
CONFIG = dict(background='w',
              foreground='k',
              imageAxisOrder='row-major')

for item in CONFIG.items():
    pg.setConfigOption(*item)


class CornerPlotItem(pg.PlotItem):

    # signal emitted after double clicking into the image
    corners_changed_signal = pyqtSignal()

    def __init__(self, *args, corners: int = 3, zero: Tuple[int] = (0, 0), **kwargs):
        super().__init__(*args, **kwargs)

        self._maximum_corners = corners
        self._zero = zero
        self._corner_plot_list: List[pg.PlotItem, ...] = []
        self._corner_coordinates: List[Tuple[float, ...]] = []
        self._angles: List[float, ...] = []

        self._omit_range_plot_list: List[pg.PlotItem, ...] = []

        self._omit_angle = 15.   # the angle that is omitted for fitting around the edge
        self.plot_omit_range = True

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> NoReturn:

        mouse_point = self.getViewBox().mapSceneToView(event.scenePos())

        x_coord = float(mouse_point.x())
        y_coord = float(mouse_point.y())

        if len(self._corner_plot_list) == self._maximum_corners:
            index = self._find_closest_index(x_coord, y_coord)
            line = self._corner_plot_list.pop(index)
            self.removeItem(line)
            self._corner_coordinates.pop(index)
            self._angles.pop(index)

        self._plot_corner(x_coord, y_coord)

    def _plot_corner(self, x_coord, y_coord):
        """Plot a corner and append the line plot to the list of lines"""

        line = self.plot([self._zero[0], float(x_coord)], [self._zero[1], float(y_coord)],
                         pen=pg.mkPen('b', width=2, style=Qt.DashLine), symbol='o', symbolBrush='b')

        self._corner_plot_list.append(line)
        self._corner_coordinates.append((x_coord, y_coord))

        phi = angle2D(self._zero, (x_coord, y_coord))
        self._angles.append(phi)

        self.corners_changed_signal.emit()

        self.show_corners()

    def _find_closest_index(self, x_new: float, y_new: float, by_distance: bool = False) -> int:
        """
        Find the closest corner when double clicking.

        :param x_new: the x coordinate of the double click
        :param y_new: the y coordinate of the double click
        :param by_distance: whether the absolute distance counts when finding the next (if True)
        or the angle (when False)
        :return: the index of the closest corner in the list of corners
        """
        if by_distance:   # take the corner where the distance is closest
            distances = np.array([((x-x_new)**2 + (y-y_new)**2)**0.5 for x, y in self._corner_coordinates])
            min_index = np.argmin(distances)
        else:  # take the corner where the angle is closest
            phi = angle2D(self._zero, (x_new, y_new))
            min_index = np.argmin(np.abs(np.array(self._angles)-phi))

        return int(min_index)

    def clear(self) -> NoReturn:
        """Remove all corners.

        Removes all corners and deletes the associated data.
        :return: NoReturn
        """

        for plot in self._corner_plot_list:
            self.removeItem(plot)
        self._corner_plot_list = []
        self._corner_coordinates = []
        self._angles = []
        self.corners_changed_signal.emit()

    @property
    def corners(self) -> List[Tuple[float]]:
        """A list of corner coordinates."""
        return self._corner_coordinates

    @property
    def angles(self) -> List[float]:
        """The list of angles corresponding to the corners."""
        return self._angles

    @property
    def omit_angle(self) -> float:
        """The angle around the edges that are omitted during the evaluation."""
        return self._omit_angle

    @omit_angle.setter
    def omit_angle(self, phi: float) -> NoReturn:
        if phi < 0 or phi > 45:
            raise ValueError(f'Invalid angle. The angle must be > 0 and < 45°.')
        self._omit_angle = phi

    @property
    def maximum_corners(self) -> int:
        """
        Number of corners that are maximally allowed.

        For instance, 3 for a Berkovich and 4 for a Vickers tips
        """
        return self._maximum_corners

    @property
    def all_corners_set(self) -> bool:
        """Indicates if all required corners were set."""
        return len(self._corner_coordinates) == self._maximum_corners

    def set_corners(self, corners: List[Tuple[float, float]]):
        """
        Plot a list of corners with the respective x and y coordinates.

        Args:
            corners: A list of corners (a tuple with the x and y coordinate of the corner)
        """

        for x, y in corners:
            self._plot_corner(x, y)

    def draw_omit_ranges(self, phi: float, length: int = 6000) -> NoReturn:
        """
        Draw the straight lines around an edge.

        These lines are drawn at +- phi.

        :param phi: the angle around the edges at which the lines will be drawn
        :param length: the length of the lines. A large number will ensure that
        the lines go beyond the image.
        :return: NoReturn
        """

        self.omit_angle = phi
        self.clear_omit_ranges()

        for phi in self._angles:
            phi1 = phi - self._omit_angle
            if phi1 < 0:
                phi1 += 360
            phi2 = phi + self._omit_angle
            if phi2 < 360:
                phi2 -= 360

            for phi in (phi1, phi2):
                x = length * np.cos(np.deg2rad(phi))
                y = length * np.sin(np.deg2rad(phi))
                self._omit_range_plot_list.append(self.plot([self._zero[0], x], [self._zero[1], y],
                                                            pen=pg.mkPen('k', width=2, style=Qt.DashDotLine)))

    def clear_omit_ranges(self) -> NoReturn:
        """
        Remove the straight lines around the edges.

        Remove the omit ranges plots, i.e. the lines around the edges that are plotted
        with +- self.omit angle.

        :return: NoReturn
        """

        for plot in self._omit_range_plot_list:
            self.removeItem(plot)

    def show_corners(self) -> NoReturn:
        """
        Show the corners as well as the straight lines connecting them with the center.

        :return: NoReturn
        """

        for plot in self._corner_plot_list:
            plot.show()

    def hide_corners(self) -> NoReturn:
        """
        Hide the corners as well as the straight lines connecting them with the center.

        :return: NoReturn
        """

        for plot in self._corner_plot_list:
            plot.hide()


class ImageAnalysisWidget(pg.GraphicsLayoutWidget):
    """This widget holds the graphs for choosing the data from the image.

    It allows for setting corners of the tips by double clicking and may show
    several different iso lines. """

    level_changed_signal = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> NoReturn:
        """
        :param parent: The parent QWidget
        """

        # we need to generate this attribute first to that we can automatically apply autoRange to self.plot.vb
        self.plot: Optional[pg.PlotItem] = None

        super().__init__(parent)

        self.image = pg.ImageItem()
        self.histogram = InvertibleHistogramItem()

        self.corner_plot: Optional[CornerPlotItem] = None

        self.iso_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen(color='r', width=2))

        self.iso_line_image = pg.IsocurveItem(pen=pg.mkPen(color='r', width=2))
        self.min_iso_line = pg.IsocurveItem(pen=pg.mkPen(color='m', width=2, style=Qt.DotLine))
        self.max_iso_line = pg.IsocurveItem(pen=pg.mkPen(color='b', width=2, style=Qt.DotLine))

        self.roi = PolyLineROI([], closed=True, pen=pg.mkPen(color='b', width=2))
        self.roi.hide()

        # Initially we set all the data to None
        self.image_data: Optional[np.ndarray] = None
        self.unit = ' - '

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        self._setup_plot()
        self._connect_signals()

        self._roi_initialized = False

    def resizeEvent(self, ev):
        """When resized the image range is automatically resized."""
        super().resizeEvent(ev)

        # when we resize the widget we automatically show all data (this is to avoid white regions around the image)
        if self.plot is not None:
            self.plot.vb.autoRange()

    def _setup_plot(self):
        """Putting the different items together."""

        box = pg.ViewBox(lockAspect=True)
        layout = self.addLayout()

        self.addItem(box)

        # the corner plot item allows adding corners by double-clicking
        self.plot = CornerPlotItem(viewBox=box)
        self.plot.addItem(self.image)
        self.plot.invertY()
        self.plot.hideButtons()   # removes the 'A' button in the lower left corner

        self.plot.showAxis('right', True)
        self.plot.showAxis('top', True)
        layout.addItem(self.plot)

        box.addItem(self.roi)

        self.histogram = InvertibleHistogramItem()
        self.histogram.setImageItem(self.image)
        self.histogram.invertY()
        self.histogram.axis.setWidth(40)

        layout.addItem(self.histogram)

        self.iso_line.setZValue(1000)

        self.histogram.vb.addItem(self.iso_line)
        self.iso_line_image.setParentItem(self.image)
        self.min_iso_line.setParentItem(self.image)
        self.max_iso_line.setParentItem(self.image)

        self._set_plot_labels()

    def _connect_signals(self):
        """Connect the signals to the slots."""

        self.iso_line.sigDragged.connect(self.update_iso_curve)
        self.histogram.sigLevelsChanged.connect(self._invert_color)
        self.histogram.sigLevelChangeFinished.connect(self._invert_color)

    def _set_plot_labels(self, unit: Optional[str] = None) -> NoReturn:
        """
        Set the labels of the plot.

        :param unit: the unit used for the data. Will be used in the image and the histogram
        :return: NoReturn
        """

        if unit is None:
            unit = self.unit

        x_label = f'x ({unit})'
        y_label = f'y ({unit})'

        for axis, label in zip(['bottom', 'top', 'left', 'right'], [x_label, x_label, y_label, y_label]):
            ax = self.plot.getAxis(axis)
            ax.setLabel(label, **{'font-size': '10pt'})
            ax.setStyle(tickTextOffset=5, tickLength=-10)

        self.histogram.axis.setLabel(f'depth ({unit})')

    def init_roi(self, scale: float = 0.25) -> NoReturn:
        """
        Initialize the roi for the first time.

        :param scale: a scale parameter that determines the initial size of the roi. Reasonable values are between
                      0.1 and 0.5
        :return: NoReturn
        """

        if self.image_data is None:
            return

        x_limits = self.plot.vb.state['limits']['xLimits']
        y_limits = self.plot.vb.state['limits']['yLimits']

        range_x = max(x_limits) - min(x_limits)  # the image plot dimension in x
        range_y = max(y_limits) - min(y_limits)  # the image plot dimension in x

        x1 = -np.arctan(np.deg2rad(30)) * scale * range_x
        x2 = np.arccos(np.deg2rad(30)) * scale * range_x
        y1 = scale * range_y

        # the editable ROI with some initial node coordinates
        self.roi.setPoints([[x1, y1], [x2, 0], [x1, -y1]])
        self.roi.show()

        self._roi_initialized = True

    def set_roi(self, points: List[Tuple[float, float]]):
        self.roi.setPoints(points)
        self.roi.show()

    def get_roi_data(self) -> Optional[np.ndarray]:
        """
        Apply the ROI to the displayed data (x, y and z).

        The data is reduced to the rectangle enclosing the ROI. Values outside the actual ROI (for instance the
        triangle) are set to np.nan.

        :return: a numpy array with the x, y and z data cropped to the chosen region. The data outside the ROI but
                 inside the enclosing rectangle is set to np.nan
        """

        if not self.roi.isVisible():
            return None

        # we can conveniently apply the roi to the x, y and z data
        data_roi_x = self.roi.getArrayRegion(self.x, self.image)
        data_roi_y = self.roi.getArrayRegion(self.y, self.image)
        data_roi_z = self.roi.getArrayRegion(self.image_data, self.image)

        # x, y, z = [x.flatten() for x in (data_roi_x, data_roi_y, data_roi_z)]

        # we need to temporarily replace the apex of the tips (which has the value) 0
        # because otherwise it would be replaced with np.nan
        data_roi_z[data_roi_z == 0] = np.inf
        data_roi_x, data_roi_y, data_roi_z = [np.where(val == 0, np.nan, val) for val in [data_roi_x, data_roi_y, data_roi_z]]
        data_roi_z[data_roi_z == np.inf] = 0.0

        return np.array([data_roi_x, data_roi_y, data_roi_z])

    def set_image_data(self, data: np.array, unit: str = '') -> NoReturn:
        """
        Supply and set the data to the image.

        :param data: Has to be an array containing the x, y, and z data. Can be either flattened or two-dimensional.
        :param unit: the unit of the data (e.g. nm)
        :return: NoReturn
        """

        self.x, self.y, self.image_data = data
        self.unit = unit

        self.image.setImage(self.image_data)
        self._invert_color()  # inverts the color of the image to match the histogram

        # we scale and move the image
        x_scale = np.abs(self.x.max() - self.x.min()) / len(np.unique(self.x))
        y_scale = np.abs(self.y.max() - self.y.min()) / len(np.unique(self.y))

        x_trans = self.x.min() / x_scale
        y_trans = self.y.min() / y_scale

        self.image.scale(x_scale, y_scale)
        self.image.translate(x_trans, y_trans)

        self.iso_line_image.setData(self.image_data)
        self.min_iso_line.setData(self.image_data)
        self.max_iso_line.setData(self.image_data)

        self.histogram.setHistogramRange(int(self.image_data.min()), int(self.image_data.max()))

        # initially we set the level of the iso line to the mean between maximum and minimum of the image data
        self.iso_line.setValue(np.mean([self.image_data.min(), self.image_data.max()]))

        # we set the maximum zoom such that no more than the actual image is shown
        self.plot.setLimits(maxYRange=self.y.max()-self.y.min(), maxXRange=self.x.max()-self.x.min(), xMin=self.x.min(),
                            xMax=self.x.max(), yMin=self.y.min(), yMax=self.y.max())
        self.plot.autoRange()

        self._set_plot_labels()

        self.update_iso_curve()

    def update_iso_curve(self, *, value: Optional[int] = None) -> NoReturn:
        """
        Updates the iso-curve in the image when the iso line in the histogram is dragged.

        The minimum and maximum draggable values are set to the minimum and maximum of the data,
        respectively.

        :param value: An integer value that is set as the new iso value. Important: this parameter is keyword only,
        i.e. it needs to be set using its name: value=<int>

        :return: NoReturn
        """

        value = value or self.iso_line.value()

        # ensure that we cannot drag the iso line out of the histogram
        if value < self.image_data.min():
            value = self.image_data.min()

        if value > self.image_data.max():
            value = self.image_data.max()

        self.iso_line.setValue(value)
        self.iso_line_image.setLevel(value)
        self.level_changed_signal.emit(int(self.iso_line.value()))

        self.show_iso_line(show=True)

    def _invert_color(self) -> NoReturn:
        """Inverts the colors of the image such that they match the histogram."""
        self.image.setLevels(self.histogram.getLevels()[::-1])

    def show_roi(self) -> NoReturn:
        """
        Show/hide the ROI.

        If the ROI has never been shown yet we initialize it first.
        :return: NoReturn
        """

        if not self._roi_initialized:
            self.init_roi()
            return
        if self.roi.isVisible():
            self.roi.hide()
        else:
            self.roi.show()

    def set_min_iso_line(self, level: int) -> NoReturn:
        """Set the level of the minimum iso line."""
        self.min_iso_line.setLevel(level)

    def show_min_iso_line(self) -> NoReturn:
        """Show the minimum iso line."""
        self.min_iso_line.show()

    def hide_min_iso_line(self) -> NoReturn:
        """Hide the minimum iso line."""
        self.min_iso_line.hide()

    def set_max_iso_line(self, level: int) -> NoReturn:
        """Set the level of the maximum iso line."""
        self.max_iso_line.setLevel(level)

    def show_max_iso_line(self) -> NoReturn:
        """Show the maximum iso line."""
        self.max_iso_line.show()

    def hide_max_iso_line(self) -> NoReturn:
        """Hide the maximum iso line."""
        self.max_iso_line.hide()

    def show_iso_line(self, *, show: Optional[bool] = None) -> NoReturn:
        """
        Display the two isolines in the image and the histogram.

        :param show: If True the lines are displayed and hidden if False.
        :return: NoReturn
        """
        if show is None:
            show = not self.iso_line.isVisible()
        if show:
            self.iso_line.show()
            self.iso_line_image.show()
        else:
            self.iso_line.hide()
            self.iso_line_image.hide()


class InvertibleHistogramItem(pg.HistogramLUTItem):
    """
    This is a small modification of the original HistogramLUTItem.

    This class allows for inverting the axis of the color scheme. By introducing a
    method called invertY. Also, the menu has been disabled for the histogram.
    """

    def __init__(self, *args, **kwargs):
        super(InvertibleHistogramItem, self).__init__(*args, **kwargs)

        self.inverted = False
        self.vb.setMenuEnabled(False)

    def paint(self, p, *args):

        pen = self.region.lines[0].pen
        rgn = self.getLevels()
        p1 = self.vb.mapFromViewToItem(self, Point(self.vb.viewRect().center().x(), rgn[0]))
        p2 = self.vb.mapFromViewToItem(self, Point(self.vb.viewRect().center().x(), rgn[1]))
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())

        for pen in [pg.mkPen((0, 0, 0, 100), width=3), pen]:
            p.setPen(pen)

            # I have changed the source code a bit here so that I can invert the y-axis and at the same time
            # avoid that the yellow lines that are being drawn to the range in the Histogram cross each other
            if self.inverted:
                p.drawLine(p2 - Point(0, 0), gradRect.bottomLeft())
                p.drawLine(p1 + Point(0, 0), gradRect.topLeft())
            else:
                p.drawLine(p1 + Point(0, 0), gradRect.bottomLeft())
                p.drawLine(p2 - Point(0, 0), gradRect.topLeft())

            p.drawLine(gradRect.topLeft(), gradRect.topRight())
            p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())

    def invertY(self, inv=True):
        self.vb.invertY(inv)
        self.inverted = inv


class PolyLineROI(pg.PolyLineROI):
    """
    A class inheriting from pg.PolyLineROI

    It is necessary to modify the original class to fix a bug with scaled images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getArrayRegion(self, data, img, axes=(0, 1), **kwds):
        """
        Return the result of ROI.getArrayRegion(), masked by the shape of the
        ROI. Values outside the ROI shape are set to 0.
        """

        # The follwoing lines were commented out because they lead to an error with scaled images

        # br = self.boundingRect()
        # if br.width() > 1000:
        #     raise Exception()

        sliced = pg.ROI.getArrayRegion(self, data, img, axes=axes, fromBoundingRect=True, **kwds)

        if img.axisOrder == 'col-major':
            mask = self.renderShapeMask(sliced.shape[axes[0]], sliced.shape[axes[1]])
        else:
            mask = self.renderShapeMask(sliced.shape[axes[1]], sliced.shape[axes[0]])
            mask = mask.T

        # reshape mask to ensure it is applied to the correct data axes
        shape = [1] * data.ndim
        shape[axes[0]] = sliced.shape[axes[0]]
        shape[axes[1]] = sliced.shape[axes[1]]
        mask = mask.reshape(shape)

        return sliced * mask

    def getAffineSliceParams(self, data, img, axes=(0, 1), fromBoundingRect=False):
        """
        Returns the parameters needed to use :func:`affineSlice <pyqtgraph.affineSlice>`
        (shape, vectors, origin) to extract a subset of *data* using this ROI
        and *img* to specify the subset.

        If *fromBoundingRect* is True, then the ROI's bounding rectangle is used
        rather than the shape of the ROI.

        See :func:`getArrayRegion <pyqtgraph.ROI.getArrayRegion>` for more information.
        """

        origin = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 0)))

        ## vx and vy point in the directions of the slice axes, but must be scaled properly
        vx = img.mapToData(self.mapToItem(img, QtCore.QPointF(1, 0))) - origin
        vy = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 1))) - origin

        lvx = np.sqrt(vx.x() ** 2 + vx.y() ** 2)
        lvy = np.sqrt(vy.x() ** 2 + vy.y() ** 2)
        ##img.width is number of pixels, not width of item.
        ##need pxWidth and pxHeight instead of pxLen ?
        sx = 1.0 / lvx
        sy = 1.0 / lvy

        vectors = ((vx.x() * sx, vx.y() * sx), (vy.x() * sy, vy.y() * sy))
        if fromBoundingRect is True:
            shape = self.boundingRect().width(), self.boundingRect().height()
            origin = img.mapToData(self.mapToItem(img, self.boundingRect().topLeft()))
            origin = (origin.x(), origin.y())
        else:
            shape = self.state['size']
            origin = (origin.x(), origin.y())

        shape = [abs(shape[0] / sx), abs(shape[1] / sy)]

        if img.axisOrder == 'row-major':
            # transpose output
            vectors = vectors[::-1]
            shape = shape[::-1]

        return shape, vectors, origin
