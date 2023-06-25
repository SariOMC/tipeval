"""
Module containing the matplotlib plot baseclass.
"""

import os
import typing as T

import matplotlib.pyplot as plt
from mayavi import mlab

from tipeval.core.typing import FilePath, Figure


_AVAILABLE_TYPES = {'matplotlib': plt.figure,
                    'mayavi': mlab.figure}

_image_count = 0  # unique identifier for the images


class BasePlot:
    """
    Base class for the plots.
    """
    def __init__(self, output_directory: T.Optional[FilePath] = None, type: str = 'matplotlib', **kwargs):
        """
        :param output_directory: the output directory where to save the image
        :param kwargs: all keyword arguments are passed to plt.figure or mlab.figure
        """
        self._type = type

        self._fig: Figure = _AVAILABLE_TYPES[self._type](**kwargs)

        self._output_directory = output_directory

    @property
    def figure(self):
        """The figure object (plt.figure, mlab.figure)."""
        return self._fig

    def save(self, file_path: T.Optional[FilePath] = None, dpi: T.Optional[int] = None,
             image_identifier: str = '') -> T.NoReturn:
        """Save the figure to a file.

        :param file_path: the path to save the figure to.
        :param dpi: the dpi used. If not given, the value given in the configuration is used.
        :param image_identifier: a string that is inserted into the file name.
        :return: NoReturn
        """

        from tipeval.config import configuration
        global _image_count

        if self._output_directory is None and file_path is None:
            self._output_directory = configuration.output_settings.default_output_folder

        if file_path is None and not image_identifier:
            _image_count += 1

        if file_path is None:
            image_identifier = image_identifier or _image_count
            file_path = os.path.join(self._output_directory, f'image_{image_identifier}'
                                     + configuration.output_settings.image_type)

        dpi = dpi or configuration.output_settings.dpi_images

        if self._type == 'matplotlib':
            self._fig.savefig(file_path, dpi=dpi)
        if self._type == 'mayavi':
            mlab.savefig(file_path)

    def __del__(self):
        if self._type == 'mayavi':
            try:
                mlab.close(scene=self.figure.scene)
            except AttributeError:
                # catching some error that occurs as a result of a mayavi-bug that always opens an additional window
                pass

    def show(self):
        """Show the figure"""

        if self._type == 'mayavi':
            mlab.show()
        if self._type == 'matplotlib':
            self.figure.show()

    def close(self):
        """Close the figure"""
        if self._type == 'mayavi':
            mlab.close()
