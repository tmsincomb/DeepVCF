"""
Where all the other files converge.
This file should not be long and should be incredabliy human readable.
People should look at this and think; wow that's simpiler than I thought. We need this in research.
"""
from typing import Callable

from pysam import AlignmentFile

from .alignment import alignment
from .biopandas import pandas as pd
from .tools import pathing


class VCFCorrectness:

    def __init__(self):
        pass

    @staticmethod
    def __methods(_class: Callable) -> dict:
        methods = {
            func: getattr(_class, func)
            for func in dir(_class)
            if callable(getattr(_class, func)) and not func.startswith("__")  # ignore dunder methods
        }
        return methods

    def __call_method(self, _class: object, option: str) -> Callable:
        methods: dict = self.__methods(alignment)
        method = methods.get(option)
        if method is None:
            raise ValueError(f'Please choose from the following options for {str(_class)}: {list(methods)}')
        return method

    def _alignment(self, option: str = 'blastn', **kwargs) -> AlignmentFile:
        method = self.__call_method(alignment, option)
        return method(**kwargs)

    def _preprocessing(self):
        pass

    def _model(self):
        pass

    def _train(self):
        pass

    def _postprocessing(self):
        pass