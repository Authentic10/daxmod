"""Provides methods to save and load objects"""
import os
import joblib


__all__ = [
    'save_object',
    'load_object'
]

def save_object(obj, name: str, folder: str = "objects"):
    """save objects
    
    Args:
        obj : the object to save
        name : the name of the object
        folder : the folder in which the object will be saved. Defaults to 'objects'
    """
    if not isinstance(name, str):
        raise TypeError
    if not isinstance(folder, str):
        raise TypeError
    objects_path = os.getcwd()
    objects_path = os.path.join(objects_path, folder)
    object_path = os.path.join(objects_path, name)
    # Create objects directory if not exists
    if not os.path.isdir(objects_path):
        os.mkdir(objects_path)   
    # Create a directory to save the object 
    if not os.path.isdir(object_path):
        os.mkdir(object_path)    
    filename = name+'.model'
    final_path = os.path.join(object_path, filename)
    print(f"Saving object at {object_path} ...")
    joblib.dump(obj, final_path)
    print(f"Object {filename} saved.")
        
      
def load_object(path:str):
    """load objects
    
    Args:
        path : the path to the object
    """
    if not isinstance(path, str):
        raise TypeError
    if not os.path.isfile(path):
        raise Exception("The file sepcified is not found")
    return joblib.load(path)