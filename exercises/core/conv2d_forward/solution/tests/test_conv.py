"""
This module contains the tests for the Conv2d
"""

# 3pps
import torch
import pytest

# Own modules
from src.conv import Conv2d


@pytest.mark.order(1)
@pytest.mark.parametrize("out_channels, kernel_size", [(10, 3), (2, 7)])
@torch.no_grad()
def test_conv(out_channels: int, kernel_size: int, inputs_conv: torch.Tensor) -> None:
    """
    This function tests the Conv2d class.

    Args:
        out_channels: output channels to the layer.
        kernel_size: kernel size for the layer.
        inputs_conv: inputs for the convolution as a fixture.

    Returns:
        None.
    """

    # Get inputs from fixture
    inputs: torch.Tensor = inputs_conv

    # Define models
    model: torch.nn.Module = Conv2d(inputs.shape[1], out_channels, kernel_size)
    model_torch: torch.nn.Module = torch.nn.Conv2d(
        inputs.shape[1], out_channels, kernel_size, bias=True, dtype=torch.double
    )

    # Set same init weights
    model.weight = model_torch.weight
    model.bias = model_torch.bias

    #
    outputs: torch.Tensor = model(inputs)
    outputs_torch: torch.Tensor = model_torch(inputs)

    # Check shape
    assert (
        outputs.shape == outputs_torch.shape
    ), f"Incorrect shape, expected {outputs_torch.shape} and got {outputs.shape}"

    # Check outputs values
    assert torch.allclose(outputs, outputs_torch), "Incorrect forward values"

    return None
