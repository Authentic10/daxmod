"Predefined classifiers"
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier


class NaiveBayes(GaussianNB):
    """Predefined Naive Bayes classifier"""

class SVM(SGDClassifier):
    """Predefined SVM classifier"""
    def __init__(self, loss="hinge", penalty="l2", alpha=1e-3, max_iter=5, tol=None,
                 random_state=0):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter,
                         tol=tol, random_state=random_state)

class MLPerceptron(MLPClassifier):
    """Predefined Multi-Layer Perceptron classifier"""
    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                 alpha=1e-4, batch_size="auto", learning_rate="constant",
                 early_stopping=True):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver=solver, alpha=alpha, batch_size=batch_size,
                         learning_rate=learning_rate, early_stopping=early_stopping)