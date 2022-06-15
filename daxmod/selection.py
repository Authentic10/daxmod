"""Provides class for feature selection"""
from sklearn.feature_selection import SelectKBest
from sklearn.utils.validation import check_is_fitted
from .constants import DEFINED_SELECT_K_FUNCTIONS
from .exceptions import SelectionFunctionError, StringVariableError
from .persistence import save_object


__all__ = [
    'SelectTopK',
]


def _get_function(function):
    " Set the function "
    function_ = function
    if isinstance(function, str):
        if function in DEFINED_SELECT_K_FUNCTIONS:
            function_ = DEFINED_SELECT_K_FUNCTIONS.get(function)
        else:
            raise SelectionFunctionError(function)
    else:
        raise StringVariableError(function)
    return function_

class SelectTopK(SelectKBest):
    "Select the top features from the data"
    def __init__(self, score_func: str ='anova', k=10):
        function_ = _get_function(function=score_func)
        super().__init__(score_func=function_, k=k)

    def save(self, name:str, folder: str = 'selectors'):
        """Save a selector

        Args:
            name (str): the name of the selector
            folder (str, optional): Folder in which to save the model. Defaults to 'selectors'.
        """
        if not isinstance(name, str):
            raise TypeError
        check_is_fitted(self)
        save_object(self, name=name, folder=folder)
         