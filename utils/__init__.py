"""Shared utility classes and functions"""
#from . import math
from .math import get_minmax_array, _iterate
#from . import inspect, math, pretty, random, skmultiflow_utils
#from .context_managers import log_method_calls
#from .data_conversion import dict2numpy, numpy2dict
#from .param_grid import expand_param_grid
#from .rolling import Rolling, TimeRolling
#from .sorted_window import SortedWindow
#from .vectordict import VectorDict

__all__ = [
    "dict2numpy",
    "expand_param_grid",
    "inspect",
    "log_method_calls",
    "math",
    "pretty",
    "numpy2dict",
    "random",
    "skmultiflow_utils",
    "Rolling",
    "SortedWindow",
    "VectorDict",
    "TimeRolling",
]

#__all__ = ["get_minmax_array"]

import numpy as np
'''
def get_minmax_array(X):
    """Utility method that returns the boundaries for each feature of the input array.
    Args:
        X (np.float array of shape (num_instances, num_features)): The input array.
    Returns:
        min (np.float array of shape (num_features,)): Minimum values for each feature in array.
        max (np.float array of shape (num_features,)): Maximum values for each feature in array.
    """
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    return min, max'''