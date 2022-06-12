"""Provide pretrained models"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from .utils import _set_activation_function, _set_classifier_units, _set_labels
from ..constants import BERT_DEFAULT

all = [
    'Bert'
]


def _bert_setup(preprocess, bert):
    """Set up of the bert preprocess and model"""    
    inputs = keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocess_ = hub.KerasLayer(preprocess, name='preprocessing')
    preprocessed_inputs = preprocess_(inputs)
    bert_ = hub.KerasLayer(bert, trainable=True, name='bert')
    outputs = bert_(preprocessed_inputs)
    net = outputs['pooled_output']
    model = keras.Model(inputs, net)
    return model

class BertLayer(keras.layers.Layer):
    """Base Layer for Bert classifier"""
    def __init__(self, preprocess, bert):
        super().__init__()
        self.classifier = _bert_setup(preprocess, bert)

    def call(self, inputs):
        return self.classifier(inputs)

class Bert(keras.Model):
    """Predefined bert model"""
    def __init__(
        self, n_classes: int, preprocess: str = BERT_DEFAULT[0],
        bert: str = BERT_DEFAULT[1], dropout: float = .1):
        if not (isinstance(n_classes, int) and n_classes > 1):
            raise ValueError(f"At least two 2 classes expected, got{n_classes}")
        super().__init__()
        self.loss_ = None
        self.metrics_ = None
        self.optimizer_ = None
        self.preds = None
        self.labels = None
        self.n_classes_ = n_classes
        self.activation = _set_activation_function(n_classes)
        self.bert = BertLayer(preprocess, bert)
        self.drop = keras.layers.Dropout(dropout)
        self.units = _set_classifier_units(n_classes)
        self.classifier = keras.layers.Dense(self.units, activation=self.activation)

    def call(self, inputs):
        outputs = self.bert(inputs)
        outputs = self.drop(outputs)
        return self.classifier(outputs)

    def auto_compile(self):
        self.loss_ = tf.keras.losses.CategoricalCrossentropy() if self.n_classes_ > 2 else tf.keras.losses.BinaryCrossentropy()
        self.metrics_ = tf.metrics.SparseCategoricalCrossentropy() if self.n_classes_ > 2 else tf.metrics.BinaryAccuracy()
        self.optimizer_ = tf.keras.optimizers.Adam(learning_rate=3e-5)
        super().compile(optimizer=self.optimizer_, loss=self.loss_, metrics=self.metrics_)

    def predict(self, X, unique_labels: list = None):
        self.labels = np.arange(self.n_classes_)
        if unique_labels is not None:
            self.labels = unique_labels
        self.preds = super().predict(X)
        return _set_labels(self.preds, self.labels)
