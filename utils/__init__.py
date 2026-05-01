"""
General utilities
"""
import os
import torch
import numpy as np
from os.path import join, exists


def print_header(stdout, style=None):
    if style is None:
        print("-" * len(stdout))
        print(stdout)
        print("-" * len(stdout))
    elif style == "bottom":
        print(stdout)
        print("-" * len(stdout))
    elif style == "top":
        print("-" * len(stdout))
        print(stdout)


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def free_gpu(tensors, delete):
    for tensor in tensors:
        tensor = tensor.detach().cpu()
        if delete:
            del tensor

