"""
This module contains utils for testing functions.
"""

# Standard libraries
import os
import random
from typing import Any

# 3pps
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix randomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def add_seed(parameters: tuple[Any, ...], num_seeds: int = 3) -> list[tuple[Any, ...]]:
    """
    This function

    Args:
        parameters: _description_
        num_seeds: _description_. Defaults to 3.

    Returns:
        _description_
    """

    return [(*parameters, seed) for seed in range(num_seeds)]
