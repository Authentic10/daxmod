"""Provides methods for data processing"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_feature_and_label_names(data: pd.DataFrame):
    """Get feature and label names of the data

    Args:
        data (pd.DataFrame): dataset 

    Raises:
        TypeError: the data is not a pandas dataframe

    Returns:
        feature(string), label(string): Returns feature and label
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError
    columns = list(data.columns)
    feature, label = columns[0], columns[-1] 
    return feature, label
       
def get_feature(data: pd.DataFrame):
    """Return the feature of the dataset

    Args:
        data (pd.DataFrame): dataset

    Raises:
        TypeError: the data is not a pandas dataframe

    Returns:
        pd.Series: pandas series
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError
    feature = list(data.columns)
    feature.pop()
    return data[feature]

def get_label(data: pd.DataFrame):
    """Return the label of the dataset

    Args:
        data (pd.DataFrame): dataset

    Raises:
        TypeError: the data is not a pandas dataframe

    Returns:
        pd.Series: pandas series
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError
    label = list(data.columns)[-1]
    return data[label]

def feature_label_split(data: pd.DataFrame):
    """Split the data into feature and label

    Args:
        data (pd.DataFrame): dataset 

    Raises:
        TypeError: the data is not a pandas dataframe

    Returns:
        feature(pd.Series), label(pd.Series): Returns the feature and label of the data
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError
    feature, label = get_feature_and_label_names(data)
    return data[feature], data[label]

def count_labels(data):
    """Count the unique labels

    Args:
        data (pd.DataFrame, pd.Series, list, np.ndarray): the dataset

    Raises:
        TypeError: the data type is none of the one mentioned above

    Returns:
        int: the number of unique labels
    """
    if not isinstance(data, (pd.DataFrame, pd.Series, list, np.ndarray)):
        raise TypeError
    if isinstance(data, pd.DataFrame):
        label = data.columns[-1]
        return len(data[label].value_counts().index)
    if isinstance(data, pd.Series):
        return len(data.value_counts())
    elif isinstance(data, (list, np.ndarray)):
        return len(list(dict.fromkeys(data)))

def encode_labels(data):
    """Return the encoded labels

    Args:
        data (pd.DataFrame, pd.Series): the data to encode

    Raises:
        TypeError: the data type is none of the one mentioned above

    Returns:
        np.ndarray: the encoded labels
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError
    le = LabelEncoder()
    if isinstance(data, pd.DataFrame):
        y = get_label(data)  
        return le.fit_transform(y)
    else:    
        return le.fit_transform(data)
      


            
             