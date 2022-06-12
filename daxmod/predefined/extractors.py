"""Provides feature extraction methods"""
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)


__all__ = [
   'BagOfWords',
   'Bigrams',
   'Ngrams',
   'TfIdf',
   'Trigrams',
]


class Ngrams(CountVectorizer):
    """Base class to create custom ngrams extractors"""
    def __init__(
        self, encoding="utf-8", stop_words=None,token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1,1), analyzer="word", binary=False):
        super().__init__(
            encoding=encoding, stop_words=stop_words,token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer, binary=binary)

class BagOfWords(Ngrams):
    """Bag of words extractor"""
    def __init__(
        self, encoding="utf-8", stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1,1), analyzer="word", binary=False):
        super().__init__(
            encoding=encoding, stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer, binary=binary)

class Bigrams(Ngrams):
    """Bigrams extractor"""
    def __init__(
        self, encoding="utf-8", stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(2, 2), analyzer="word", binary=False):
        super().__init__(
            encoding=encoding, stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer, binary=binary)   

class Trigrams(Ngrams):
    """Trigrams extractor"""
    def __init__(
        self, encoding="utf-8", stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(3, 3), analyzer="word", binary=False):
        super().__init__(
            encoding=encoding, stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, analyzer=analyzer, binary=binary)

class TfIdf(TfidfVectorizer):
    """Tf and TfIdf extractor"""
