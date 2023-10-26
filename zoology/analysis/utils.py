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