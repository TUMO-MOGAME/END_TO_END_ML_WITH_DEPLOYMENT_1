import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import ProjectException

def save_object(file_path, obj):
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception in case of an error
        raise ProjectException(e, sys)
