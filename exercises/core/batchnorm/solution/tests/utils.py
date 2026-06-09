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
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix randomness.
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
    Attach a few seed values to the provided parameters.
    """

    return [(*parameters, seed) for seed in range(num_seeds)]
