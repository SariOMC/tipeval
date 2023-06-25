import abc
from datetime import date
import os
import typing as T

import h5py
import numpy as np

from tipeval.core.typing import FilePath
from tipeval.core.utils.data import parse_time


def make_output_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def _save_load_hdf(file_handle: FilePath, data: np.array, meta: T.Dict[str, T.Any]) -> T.NoReturn:

    with h5py.File(file_handle, 'a') as f:
        f.attrs['input_file'] = os.path.split(meta['input_file'])[-1]
        f.attrs['imaging_method'] = meta['imaging_method']
        f.attrs['recording_date'] = parse_time(meta['date'])

        f.attrs['tip_type'] = meta['tip_type']
        f.attrs['tip_name'] = meta['tip_name']
        if 'data' in f:
            del f['data']
        f['data'] = data
        f['data'].attrs['unit'] = meta['new_unit']


def _save_choose_data_hdf(file_handle: FilePath, *args, **kwargs):

    """
            info = dict(
            roi=self._get_roi(),
            min=float(self.min_depth_field.text() or 0.),
            max=float(self.max_depth_field.text() or np.inf),
            angle_around_edges=float(self.angle_box.value()),
            corners=self.image_widget.plot.corners,
            corner_angles=self.image_widget.plot.angles
        )
    """

    with h5py.File(file_handle, 'a') as f:
        ...


def _save_fit_data(file_handle: FilePath, *args, **kwargs):
    ...
    with h5py.File(file_handle, 'a') as f:
        ...


def _make_output_file(directory: FilePath, info: T.Dict[str, T.Any]):
    now = date.today().strftime("%d-%m-%Y")
    output_file = f"evaluation_{info['tip_name']}_{now}.hdf5"
    output_file = os.path.join(directory, output_file)
    return output_file


class StorageMixin(abc.ABC):

    @abc.abstractmethod
    def save_state(self, state):
        pass

    @abc.abstractmethod
    def load_state(self):
        pass


class HDFStorage(StorageMixin):

    # the keys correspond to the values in the Evaluation.State enum
    save_state_functions = {1: _save_load_hdf,
                            2: _save_choose_data_hdf,
                            3: _save_fit_data}

    load_state_functions = {}

    def __init__(self, info: T.Dict[str, T.Any]) -> None:
        output_directory = info['output_directory']
        make_output_directory(output_directory)
        self.output_file = _make_output_file(output_directory, info)

    def save_state(self, state: int, *args, **kwargs) -> T.NoReturn:
        self.save_state_functions[state](self.output_file, *args, **kwargs)

    def load_state(self):
        pass
