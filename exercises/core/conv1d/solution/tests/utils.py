"""
This module contains auxiliary code for the tests.
"""

# Standard libraries
import os
import random
from typing import Any

# 3pps
import torch


def set_seed(seed: int) -> None:
    """
    Set a seed and ensure deterministic behavior.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def add_seed(parameters: tuple[Any, ...], num_seeds: int = 3) -> list[tuple[Any, ...]]:
    """
    Append seeds to a parameter tuple.
    """

    return [(*parameters, seed) for seed in range(num_seeds)]
