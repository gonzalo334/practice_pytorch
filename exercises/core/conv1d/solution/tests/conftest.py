"""
This module contains the code for the fixtures.
"""

# 3pps
import pytest
import torch

# Own modules
from tests.utils import add_seed, set_seed


@pytest.fixture(params=[*add_seed((8, 32, 16)), *add_seed((4, 12, 8))])
def inputs_1d(request) -> torch.Tensor:
    """
    This function defines example random inputs.

    Args:
        request: Argument containing the introduced arguments.

    Returns:
        Inputs tensor. Dimensions: [batch, channels, length].
    """

    batch_size, num_channels, length, seed = request.param
    set_seed(seed)
    return torch.rand(batch_size, num_channels, length).uniform_(-10, 10).double()
