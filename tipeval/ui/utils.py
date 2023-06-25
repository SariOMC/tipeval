from functools import wraps
import typing as T

from PyQt5.QtWidgets import QWidget


def block_signals(func):
    """
    Turn off signal emission for a dict of QWidgets.

    The function to be decorated has to have a keyword argument named
    widgets, which has to be a dictionary where all values are QWidgets.
    Signal emission is turned on again after the function call.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """The function wrapped by block_signal_decorator"""
        widgets = kwargs.get('widgets', {})

        for widget in widgets.values():
            widget.blockSignals(True)

        result = func(*args, **kwargs)

        for widget in widgets.values():
            widget.blockSignals(False)

        return result
    return wrapper
