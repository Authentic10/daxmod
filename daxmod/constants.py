"""Provides constants"""
from sklearn.feature_selection import f_classif, chi2
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, balanced_accuracy_score)
from .predefined.classifiers import MLPerceptron, NaiveBayes, SVM
from .predefined.extractors import BagOfWords, Bigrams, TfIdf, Trigrams


"Predefined extractors"
DEFINED_EXTRACTORS = {
    'bow' : BagOfWords(),
    'bigrams': Bigrams(),
    'trigrams': Trigrams(),
    'tf_idf': TfIdf(),
    'tf': TfIdf(use_idf=False),
}

"Predefined classifiers"
DEFINED_CLASSIFIERS = {
    'naive_bayes' : NaiveBayes(),
    'svm' : SVM(),
    'mlp' : MLPerceptron(),
}

"Predefined selection functions"
DEFINED_SELECT_K_FUNCTIONS = {
    'anova' : f_classif,
    'chi2' : chi2
}

"Default bert model links"
BERT_DEFAULT = (
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2' 
)

"Threshold for predictions"
BINARY_THRESH = 0.5

"Metrics"
METRICS = {
    'accuracy' : accuracy_score,
    'balanced_accuracy' : balanced_accuracy_score,
    'f1' : f1_score,
    'precision' : precision_score,
    'recall' : recall_score
}
