"""
This module contains the class for saving the evaluation when the GUI is used.
"""
from __future__ import annotations

import enum
import collections.abc
from functools import wraps
import typing as T

import numpy as np

from tipeval.core.geometries import Plane
from tipeval.core.utils.data import load_data, get_data_for_fit
from tipeval.core.utils.storage import HDFStorage, StorageMixin
from tipeval.core.tips import TIP_CLASSES, Tip



def ensure_iterable(o: T.Any, default_type=tuple) -> T.Iterable:
    if isinstance(o, dict):
        return o,
    if isinstance(o, collections.abc.Iterable):
        return o
    else:
        return default_type([o])


def task_state(state):
    def decorate(f):
        @wraps(f)
        def wrapped(obj, *args, **kwargs):
            result = ensure_iterable(f(obj, *args, **kwargs))
            obj._internal_state[state] = result
            obj.state = state
            obj.storage.save_state(state.value, *result if result is not None else ())
            return result
        return wrapped
    return decorate


class Evaluation:

    class State(enum.Enum):
        LoadFile = 1
        ChooseData = 2
        FitArea = 3
        DetermineRadius = 4
        SaveAreaFunction = 5

    def __init__(self) -> None:
        self._internal_state = {}
        self._state = None
        self.state_changed = []

        self.storage: T.Optional[StorageMixin] = None

        self.tip: T.Optional[Tip] = None

        self.fit_data: T.Optional[np.array] = None

    @task_state(State.LoadFile)
    def load_file(self, info: T.Dict[str, T.Any]):
        if self.storage is None:
            self.storage = HDFStorage(info)
        data_converted = load_data(info)
        return data_converted, info

    @task_state(State.ChooseData)
    def choose_data(self, info) -> T.Tuple[T.Dict[str, T.Any], Tip, np.array, T.Dict[str, T.Any]]:
        roi_data, info_load = self.result(self.State.LoadFile)
        tip_type = info_load['tip_type']
        unit = info_load['new_unit']
        name = info_load['tip_name']

        roi_data = info['roi'][0] if info['roi'][0] is not None else roi_data

        self.fit_data = get_data_for_fit(roi_data, angles=info['corner_angles'], minimum=info['min'], maximum=info['max'],
                                    omit_angle=info['angle_around_edges'])

        Tip = TIP_CLASSES.get(tip_type, None)

        if Tip is None:
            raise NotImplementedError(f'Tip {tip_type} is not yet implemented!')

        self.tip = Tip([Plane.from_dataset(*face) for face in self.fit_data], unit=unit, name=name)

        return info_load, self.tip, self.fit_data, info

    @task_state(State.FitArea)
    def fit_area(self, *, contact_depths: T.Optional[np.ndarray] = None):
        *_, info = self.result(Evaluation.State.ChooseData)
        max_ = info['max']
        data, *_ = self.result(Evaluation.State.LoadFile)

        print(data)

    @task_state(State.DetermineRadius)
    def determine_radius(self):
        ...

    @task_state(State.SaveAreaFunction)
    def save_area_function(self):
        ...

    @property
    def state(self):
        return self._state

    def emit_state_changed(self):
        for slot in self.state_changed:
            slot(self._internal_state)

    @state.setter
    def state(self, s):
        if s != self._state:
            self._state = s

    def result(self, state: Evaluation.State):
        return self._internal_state.get(state, None)




