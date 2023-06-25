"""This module contains some type shortcuts used in the project"""

import os
import typing as T

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np

FilePath = T.TypeVar('FilePath', bound=T.Union[str, bytes, os.PathLike])

ImageLimit = T.Optional[T.Union[float, T.Tuple[float]]]

Type = T.TypeVar('Type')

RGB = T.Union[T.List[float], T.Tuple[float], str]

Point2D = T.Union[T.List[float], T.Tuple[float, ...]]

Depth = T.Union[float, np.ndarray]

Figure = T.Optional[T.Union[plt.figure, mlab.figure]]
