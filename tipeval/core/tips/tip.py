"""
Base tip class for all different classes of tips
"""
import abc
import typing as T

from tipeval.core.utils.data import save_tip_to_hdf


class Tip(abc.ABC):

    @property
    @abc.abstractmethod
    def type(self):
        pass

    @abc.abstractmethod
    def save_area_function(self):
        pass

    @abc.abstractmethod
    def calculate_cross_sections(self):
        pass

    @classmethod
    @abc.abstractmethod
    def from_file(cls):
        pass

    def save_to_file(self) -> T.NoReturn:
        """
        Save the tip to the attached hdf5 file.

        Returns:
            NoReturn
        """
        if hasattr(self, '_file'):
            save_tip_to_hdf(self, self._file)
        else:
            raise AttributeError(f"The tip {self} does not have an hdf5-File attached.")
