"""Provides custom exceptions"""

class ClassifierNotFoundError(Exception):
    """Classifier not found error"""
    def __init__(self, classifier, message="Classifier not found. Possible values are:\
                 'naive_bayes', 'svm', and 'mlp' ") -> None:
        self.classifier = classifier
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.classifier} -> {self.message}'

class ExtractorNotFoundError(Exception):
    """Exractor not found error"""
    def __init__(self, extractor, message="Extractor not found. Possible values are:\
                 'bow', 'bigrams', 'trigrams', 'tf', and 'tf_idf' ") -> None:
        self.extractor = extractor
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.extractor} -> {self.message}'

class SklearnClassifierError(Exception):
    """SKlearn classifier not found error"""
    def __init__(self, classifier, message="The classifier variable must be a string\
            or an Sklearn classifier.") -> None:
        self.classifier = classifier
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.classifier} -> {self.message}'

class SelectionFunctionError(Exception):
    """Selection function not found error"""
    def __init__(self, function, message="Selection method not found. Possible values are: \
                    'anova' and 'chi2'.") -> None:
        self.function = function
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.function} -> {self.message}'

class StringVariableError(Exception):
    """String variable not found error"""
    def __init__(self, variable, message="The variable must be a string") -> None:
        self.variable = variable
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.variable} -> {self.message}'

class TrainFolderNotFoundError(Exception):
    """Train folder not found error"""
    def __init__(self, message="The folder must have a subfolder named train") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class NotDirectoryError(Exception):
    """Not a directory error"""
    def __init__(self, path, message="The full path specified is not a directory") -> None:
        self.path = path
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.path} -> {self.message}'

class NumberOfClassesError(Exception):
    """Number of classes error"""
    def __init__(self, classes, message="Your folder must have at least 2 classes") -> None:
        self.classes = classes
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.classes} -> {self.message}'

class NoFilesError(Exception):
    """No files found error"""
    def __init__(self, folder, message="There are no files in the folder") -> None:
        self.folder = folder
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.folder} -> {self.message}'

class MetricError(Exception):
    """Metric not found error"""
    def __init__(self, metric, message="The selected score is not available. Try 'accuracy' or 'balanced_accuracy',\
                             or 'f1' or 'precision' or 'recall' instead") -> None:
        self.metric = metric
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.metric} -> {self.message}'