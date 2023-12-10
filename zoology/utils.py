
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