"""
This module contains the code for testing the GroupNorm.
"""

# 3pps
import torch
import pytest

# Own libraries
from src.group_norm import GroupNorm


@pytest.mark.order(1)
@pytest.mark.parametrize("num_groups", [2, 3])
def test_group_norm(num_groups: int, inputs: torch.Tensor) -> None:
    """
    This function is to test the GroupNorm.

    Returns:
        None.
    """

    # Define modules
    module: torch.nn.Module = GroupNorm(num_groups, inputs.shape[1])
    module_torch: torch.nn.Module = torch.nn.GroupNorm(
        num_groups, inputs.shape[1], affine=False
    )

    # Compute outputs
    outputs: torch.Tensor = module(inputs)
    outputs_torch: torch.Tensor = module_torch(inputs)

    # Check shape
    assert outputs.shape == outputs_torch.shape, (
        f"Incorrect output shape, expected {outputs_torch.shape} and got "
        f"{outputs.shape}"
    )

    # Check outputs value
    assert torch.allclose(outputs, outputs_torch), "Incorrect outputs value"

    return None
