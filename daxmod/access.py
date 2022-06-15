""" Provides methods to load data"""
import os
from joblib import Parallel, delayed
import pandas as pd
from .exceptions import (TrainFolderNotFoundError, NotDirectoryError, 
                         NumberOfClassesError, NoFilesError)

__all__ = [
 'load_from_file',
 'load_from_folder',
]


def _loader(path:str, file:str, encoding:str):
    file_ = os.path.join(path, file)
    if file_.endswith('.txt'):
        with open(file_, encoding=encoding) as f:
            content = f.read()
    return content

def load_from_file(filepath:str, header='infer', sep: str = ',',
                   encoding=None):
    """Load data from file

    Args:
        filepath (str): the file to load.
        header (str, optional): the header of the file. Defaults to 'infer'.
        sep (str, optional): the file separtor. Defaults to ','.
        encoding (_type_, optional): the encoding of the file. Defaults to None.

    Returns:
        pd.DataFrame: dataframe created from the files
    """
    data_frame = pd.read_csv(filepath_or_buffer=filepath, header=header,
                             sep=sep, encoding=encoding)
    return data_frame


def load_from_folder(dataset_folder:str, sub: str = 'train',
                     encoding='utf-8', n_jobs: int = None):
    """load data from dataset

    Args:
        dataset_folder (str): the dataset folder
        sub (str, optional): the subfolder to load. Defaults to 'train'.
        encoding (str, optional): encoding of the files. Defaults to 'utf-8'.
        n_jobs: (int, optional): the number of processors to use. Defaults to None which 
        use one processor. To use all processors available, use -1.
    Raises:
        TrainFolderNotFoundError: the train folder is not found
        NotDirectoryError: the full path specified is not a directory
        NumberOfClassesError: the folder does not have 2 files at minimum
        NoFilesError: There is no file fond in the folder

    Returns:
        pd.DataFrame: dataframe created from the files
    """
    folder_path = ''
    if dataset_folder is None:
        raise Exception("Please, specify the dataset folder")
    if sub is None:  # When the sub parameter is not specified
        f_list = os.listdir(dataset_folder)
        if 'train' in f_list:  # Look for the train folder
            folder_path = os.path.join(dataset_folder, 'train')
        else:
            raise TrainFolderNotFoundError
    else:
        folder_path = os.path.join(dataset_folder, sub)
        if not os.path.isdir(folder_path):
            raise NotDirectoryError(folder_path)
    folder_list = os.listdir(folder_path)
    classes = len(folder_list)
    classes_info = {}
    if classes >= 2:  # The dataset must have at least two classes
        for class_ in folder_list:
            classes_info[class_] = os.path.join(folder_path, class_)
    else:
        raise NumberOfClassesError(classes)
    data_frame = pd.DataFrame(columns=['text','label'])
    for key, folder in classes_info.items():
        list_ = os.listdir(folder)
        if len(list_) > 0:
            output = Parallel(n_jobs=n_jobs)(delayed(_loader)(folder, file, encoding) for file in list_)
            data_frame = data_frame.append(pd.DataFrame({'text':output,'label':key}))
        else:
            raise NoFilesError(folder)
    return data_frame.sample(frac=1).reset_index(drop=True)

        
    