
from typing import Callable, Union
import random 
import numpy as np 
import torch


def import_from_str(path) -> Union[type, Callable]:
    """
    Imports a class or function from a string. 
    Format: "module.submodule.ClassName" or "module.submodule.function_name".

    Args: 
        path (str): The string representing the class or function to import.
    """
    module_name, obj_name = path.rsplit(".", 1)
    try:
        module = __import__(module_name, fromlist=[obj_name])
        obj = getattr(module, obj_name)
        return obj
    except ImportError as e:
        raise ImportError(f"Module '{module_name}' not found.") from e
    except AttributeError as e:
        raise AttributeError(f"Class '{obj_name}' not found in module '{module_name}'.") from e

def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


from typing import List

import wandb
import pandas as pd


def flatten(data: dict, parent_key:str=None, sep: str='.'):
    """
    Flatten a multi-level nested collection of dictionaries and lists into a flat dictionary.
    
    The function traverses nested dictionaries and lists and constructs keys in the resulting 
    flat dictionary by concatenating nested keys and/or indices separated by a specified separator.
    
    Parameters:
    - data (dict or list): The multi-level nested collection to be flattened.
    - parent_key (str, optional): Used in the recursive call to keep track of the current key 
                                  hierarchy. Defaults to an empty string.
    - sep (str, optional): The separator used between concatenated keys. Defaults to '.'.
    
    Returns:
    - dict: A flat dictionary representation of the input collection.
    
    Example:
    
    >>> nested_data = {
    ...    "a": 1,
    ...    "b": {
    ...        "c": 2,
    ...        "d": {
    ...            "e": 3
    ...        }
    ...    },
    ...    "f": [4, 5]
    ... }
    >>> flatten(nested_data)
    {'a': 1, 'b.c': 2, 'b.d.e': 3, 'f.0': 4, 'f.1': 5}
    """
    items = {}
    if isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key is not None else str(i)
            items.update(flatten(v, new_key, sep=sep))
    elif isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key is not None else k
            items.update(flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items


def unflatten(d: dict) -> dict:
    """ 
    Takes a flat dictionary with '/' separated keys, and returns it as a nested dictionary.
    
    Parameters:
    d (dict): The flat dictionary to be unflattened.
    
    Returns:
    dict: The unflattened, nested dictionary.
    """
    import numpy as np
    result = {}

    for key, value in d.items():
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        
        if (isinstance(value, (np.float64, np.float32, float)) and np.isnan(value)):
            # need to check if value is nan, because wandb will create a column for every
            # possible value of a categorical variable, even if it's not present in the data
            continue
        
        d[parts[-1]] = value


    # check if any dicts have contiguous numeric keys, which should be converted to list
    def convert_to_list(d):
        if isinstance(d, dict):
            try:
                keys = [int(k) for k in d.keys()]
                keys.sort()
                if keys == list(range(min(keys), max(keys)+1)):
                    return [d[str(k)] for k in keys]
            except ValueError:
                pass
            return {k: convert_to_list(v) for k, v in d.items()}
        return d

    return convert_to_list(result)


def fetch_wandb_runs(project_name: str, filters: dict=None, **kwargs) -> pd.DataFrame:
    """
    Fetches run data from a W&B project into a pandas DataFrame.
    
    Parameters:
    - project_name (str): The name of the W&B project.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the run data.
    """
    # Initialize an API client
    api = wandb.Api()
    
    filters = {} if filters is None else filters
    for k, v in kwargs.items():
        if isinstance(v, List):
            filters[f"config.{k}"] = {"$in": v}
        else:
            filters[f"config.{k}"] = v
    
    # Get all runs from the specified project (and entity, if provided)
    runs = api.runs(
        project_name,
        filters=filters
    )
    
    # Create a list to store run data
    run_data = []

    # Iterate through each run and extract relevant data
    for run in runs:
        
        data = {
            "run_id": run.id,
            "name": run.name,
            "project": run.project,
            "user": run.user.name,
            "state": run.state,
            **flatten(run.config),
            **flatten({**run.summary})
        }
        run_data.append(data)
    
    # Convert list of run data into a DataFrame
    df = pd.DataFrame(run_data)
    
    df = df.dropna(axis="columns", how="all")

    # can't be serialized
    if "_wandb" in df.columns:
        df = df.drop(columns=["_wandb"])
    if "val_preds" in df.columns:
        df = df.drop(columns=["val_preds"])

    return df