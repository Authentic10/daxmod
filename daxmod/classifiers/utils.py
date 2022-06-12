"""Provide functions for Keras models"""
import numpy as np
from ..constants import BINARY_THRESH

def _set_classifier_units(n_classes):
    """Set the units of the dense layer"""
    if n_classes == 2:
        return 1
    elif n_classes > 2:
        return n_classes

def _set_activation_function(n_classes):
    """Set the activation function for the dense layer"""
    if n_classes == 2:
        return 'sigmoid'
    elif n_classes > 2:
        return 'softmax'

def _set_labels(predictions, classes):
    """Set the labels after the predictions"""
    labels = []
    if len(classes) == 2:
        labels = np.where(predictions > BINARY_THRESH, 1, 0)
    elif len(classes) > 2:
        labels = list(map(lambda x: classes[np.argmax(x)], predictions))
    return np.reshape(labels, -1)
