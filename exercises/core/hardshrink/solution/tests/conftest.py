"""
This module contains the test fixtures.
"""

# 3pps
import torch
import pytest

# Own modules
from tests.utils import add_seed, set_seed


@pytest.fixture(params=[*add_seed((64, 6, 32, 32)), *add_seed((32, 12, 32, 16))])
def inputs(request) -> torch.Tensor:
    """
    This function is a fixture to define example random inputs.

    Args:
        request: Argument containing the introduced arguments.

    Returns:
        Inputs tensor. Dimensions: [batch, channels, height, width].
    """

    # Get parameters
    batch_size: int
    num_channels: int
    height: int
    width: int
    seed: int
    batch_size, num_channels, height, width, seed = request.param

    # Set seed
    set_seed(seed)

    # Define inputs
    inputs: torch.Tensor = (
        torch.rand(batch_size, num_channels, height, width).uniform_(-10, 10).double()
    )

    return inputs


@pytest.fixture(params=[(64, 6, 32, 32), (32, 12, 32, 16)])
def inputs_zero(request) -> torch.Tensor:
    """
    This function is a fixture to define example zero inputs.

    Args:
        request: Argument containing the introduced arguments.

    Returns:
        Inputs tensor. Dimensions: [batch, channels, height, width].
    """

    # Get parameters
    batch_size: int
    num_channels: int
    height: int
    width: int
    batch_size, num_channels, height, width = request.param

    # Define inputs
    inputs: torch.Tensor = torch.zeros(batch_size, num_channels, height, width).double()

    return inputs
