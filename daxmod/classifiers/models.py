"""Provides class for training and evaluating classifiers"""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from ..constants import  DEFINED_CLASSIFIERS, METRICS
from ..exceptions import ClassifierNotFoundError, SklearnClassifierError, MetricError
from ..persistence import save_object


__all__ = [
    'Models',
]


def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )

def _get_classifier(classifier):
    """Get the classifier from the predefined

    Keyword arguments:
    classifier -- (string) the classifier to search for
    Return: (Sklearn classifier) the classifier if available
    """

    classifier_ = classifier
    if isinstance(classifier, str) or is_classifier(classifier):
        if isinstance(classifier, str):
            if classifier in DEFINED_CLASSIFIERS:
                classifier_ = DEFINED_CLASSIFIERS.get(classifier)
            else:
                raise ClassifierNotFoundError(classifier_)
        else:
            classifier_ = classifier
    else:
        raise SklearnClassifierError(classifier_)
    return classifier_

class Models(BaseEstimator, ClassifierMixin):
    """Train and evaluate models"""

    def __init__(self, classifier='naive_bayes'):
        """Initialization with a classifier

        Args:
            classifier (str, optional): the classifier to use. Possible values are 'naive_bayes', 'svm', and 'mlp'. Defaults to 'naive_bayes'.
        """
        self.classifier_ = None
        self.classifier = _get_classifier(classifier)

    def fit(self, X, y):
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y)
        return self

    @available_if(_classifier_has("predict"))
    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    @available_if(_classifier_has("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.classifier_.predict_proba(X)

    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier_.decision_function(X)

    def evaluate(self, X, y, metric: str = 'accuracy', sample_weight=None):
        """Get the score of the model on a dataset

        Args:
            X (array-like): Features of the dataset
            y (array-like): Targets of the dataset
            metric (str, optional): Metric on which to score the model. Defaults to 'accuracy'.
            sample_weight (array-like, optional): Sample weights. Defaults to None.

        Raises:
            MetricError: Returns an error when the metric does not exist

        Returns:
            float: Return the score selected
        """
        if metric not in METRICS:
            raise MetricError(metric)
        check_is_fitted(self)
        score_ = METRICS.get(metric)
        return score_(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight)

    def save(self, name:str, folder: str = 'models'):
        """Save a model

        Args:
            name (str): the name of the model
            folder (str, optional): Folder in which to save the model. Defaults to 'models'.
        """
        if not isinstance(name, str):
            raise TypeError
        check_is_fitted(self)
        save_object(self, name=name, folder=folder)
    