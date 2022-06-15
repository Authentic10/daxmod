"""Provides feature extraction methods"""
from sklearn.base import BaseEstimator, clone, TransformerMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from .constants import DEFINED_EXTRACTORS
from .exceptions import ExtractorNotFoundError, StringVariableError
from .persistence import save_object


__all__ = [
    'Extractors',
]


def _extractor_has(attr):
    """Check if we can delegate a method to the underlying extractor.

    First, we check the first fitted extractor if available, otherwise we
    check the unfitted extractor.
    """
    return lambda estimator: (
        hasattr(estimator.extractor_, attr)
        if hasattr(estimator, "extractor_")
        else hasattr(estimator.extractor, attr)
    )

def _get_extractor(extractor):
    """Select the extractor"""
    extractor_ = extractor
    if isinstance(extractor, str):
        if extractor in DEFINED_EXTRACTORS:
            extractor_ = DEFINED_EXTRACTORS.get(extractor)
        else:
            raise ExtractorNotFoundError(extractor)
    else:
        raise StringVariableError(extractor)
    return extractor_

class Extractors(BaseEstimator, TransformerMixin):
    """Extract features from data"""
    def __init__(self, extractor='bow'):
        """Initialization with an extractor

        Args:
            extractor (str, optional): the extractor to use. Possible values are 'bow', 'bigrams',
            'trigrams', 'tf_idf', and 'tf'. Defaults to 'bow'.
        """
        self.extractor_ = None
        self.extractor = _get_extractor(extractor)

    def fit(self, X, y):
        self.extractor_ = clone(self.extractor)
        self.extractor_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.extractor_.transform(X)

    def fit_transform(self, X, y):
        self.extractor_ = clone(self.extractor)
        self.extractor_.fit(X, y)
        return self.extractor_.transform(X)

    @available_if(_extractor_has("get_feature_names_out"))
    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.extractor_.get_feature_names_out()

    def save(self, name:str, folder: str = 'extractors'):
        """Save an extractor

        Args:
            name (str): the name of the extractor
            folder (str, optional): Folder in which to save the extractor. Defaults to 'extractors'.
        """
        if not isinstance(name, str):
            raise TypeError
        check_is_fitted(self)
        save_object(self, name=name, folder=folder)