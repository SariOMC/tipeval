"""This module contains some of the utility functions used for displaying the image data/results."""

from __future__ import annotations

import typing as T

from mayavi import mlab
import numpy as np
from plotly.graph_objects import Figure, Scatter3d, Surface
import plotly
from scipy.ndimage import sobel

from tipeval.core.geometries.point import Point
from tipeval.core.typing import RGB


# these are the default colors used by plotly
DEFAULT_COLORS = plotly.colors.qualitative.Plotly


if T.TYPE_CHECKING:
    from tipeval.core.tips import Tip, ThreeSidedPyramidTip


def sobel_filter(image: np.ndarray, **kwargs) -> np.array:
    """
    Apply a Sobel filter to the image.

    The filter is applied along both, the x and y directions.

    :param image: the image data, has to be a 2D numpy array
    :param kwargs: key word arguments passed to scipy.ndimage.sobel
    :return: a numpy array with the same shape as image
    """
    return np.hypot(sobel(image, axis=1, **kwargs), sobel(image, axis=0, **kwargs))


def plot_between_points(point1: Point, point2: Point, thickness: float, color: RGB,
                        figure: T.Any = None, **kwargs) -> T.NoReturn:
    """
    Plot a line between two instances of Point using mayavi.

    :param point1: the first point
    :param point2: the second point
    :param thickness: the thickness of the line
    :param color: the line color
    :param figure: the figure into which to plot the line
    :param kwargs: keyword arguments passed to mlab.plot3d
    :return: NoReturn
    """

    x, y, z = np.array([point1.to_array(), point2.to_array()]).T
    mlab.plot3d(x, y, -z, tube_radius=thickness, color=color, figure=figure, **kwargs)


def plot_between_points_plotly(point1: Point, point2: Point, **kwargs) -> plotly.graph_objects.Scatter3d:
    """
    Draw a line between two points using plotly.

    :params point1, point2: the two Point objects between which to draw a line.
    :return: the Scatter3D object of the line.
    """

    x, y, z = np.array([point1.to_array(), point2.to_array()]).T
    return Scatter3d(x=x, y=y, z=-z, line=dict(color='black'), mode='lines', **kwargs)


def plot_frame_plotly(tip: ThreeSidedPyramidTip, max_z: float = 150) -> T.List[plotly.graph_objects.Scatter3d]:
    """
    Plot the fitted tip.

    This function uses plotly's Scatter3D objects to produce the fit as some kind of wire frame.

    :param tip: the tip object for which to plot the frame.
    :param max_z: the depth value up to which the tip should be drawn.
    :return: a list of Scatter3D objects.
    """

    points = []
    plots = []

    for edge in tip.edges:
        point = edge.point_on_line(edge.get_t_for_z_value(max_z))
        points.append(point)
        plots.append(plot_between_points_plotly(point, tip.apex, name='frame', showlegend=False, legendgroup='frame'))

    for i, point in enumerate(points):
        plots.append(plot_between_points_plotly(point, points[i - 1], name='frame', legendgroup='frame',
                                                showlegend=i == 0))

    return plots


def plot_fit_faces(tip: ThreeSidedPyramidTip, fit_data: np.array) -> T.List[plotly.BaseFigure]:
    """
    Plot the fit of the faces of a tip using plotly.

    This function plots the data using Scatter3D and the fit using Surface.

    :param tip: an instance of a tip object.
    :param fit_data: the data used for fitting the faces. Should be an array
    containing an array of xyz data for each face.
    :return: a list of the plot objects.
    """

    plots = []

    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

    for x, y, z in fit_data:
        min_x = np.min(x) if np.min(x) < min_x else min_x
        max_x = np.max(x) if np.max(x) > max_x else max_x
        min_y = np.min(y) if np.min(y) < min_y else min_y
        max_y = np.max(y) if np.max(y) > max_y else max_y

    for i, (plane, (x, y, z), color) in enumerate(zip(tip.faces, fit_data, DEFAULT_COLORS)):

        xx, yy, zz = plane.z_from_xy(min_x, max_x, min_y, max_y, 2)

        plots.append(Scatter3d(x=x, y=y, z=-z, mode='markers', name=f'Face {i}', legendgroup=f'Face {i}',
                               marker=dict(color=color)))

        plots.append(Surface(x=xx, y=yy, z=-zz, showscale=False, name=f'Face {i}', legendgroup=f'Face {i}',
                             opacity=0.5, colorscale=[[0, color], [1, color]], showlegend=True,
                             contours=dict(x=dict(highlight=False),
                                           y=dict(highlight=False),
                                           z=dict(highlight=False))))

    return plots


def fit_plot_plotly(tip: Tip, fit_data: np.array) -> T.Tuple[Figure, str]:
    """
    Plot the fit of a dataset with a tip object using plotly.

    This function plots the data together with the fit of the faces
    and the total fitted tip using plotly. It returns an html text that
    can subsequently be used, for instance in a PyQt QWebEngineView.

    :param tip: an instance of a tip object.
    :param fit_data: the data used for fitting the faces. Should be an array
    containing an array of xyz data for each face.
    :return: html text of the plotly plot.
    """

    plots = []

    from tipeval.core.tips import ThreeSidedPyramidTip
    if isinstance(tip, ThreeSidedPyramidTip):
        plots.extend(plot_fit_faces(tip, fit_data))
        plots.extend(plot_frame_plotly(tip))

    # create the plotly figure
    fig = Figure(plots, layout=dict(scene=dict(xaxis=dict(showspikes=False),
                                               yaxis=dict(showspikes=False),
                                               zaxis=dict(showspikes=False))))
    fig.update_traces(
        hovertemplate=None,
        hoverinfo='skip'
    )

    # we create html code of the figure
    html = '<html><body>'
    html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
    html += '</body></html>'

    return fig, html
