"""
This module contains the tests for the Conv2d
"""

import torch
import pytest

from src.conv import Conv2d



@pytest.mark.order(1)
@pytest.mark.parametrize("out_channels, kernel_size", [(5, 4), (7, 6)])
@torch.no_grad()
def test_conv_forward(out_channels, kernel_size):

    B, Cin, H, W = 3, 4, 15, 13

    inputs = torch.randn(
        B, Cin, H, W,
        dtype=torch.double,
    )

    model = Conv2d(Cin, out_channels, kernel_size)
    model_torch = torch.nn.Conv2d(
        Cin,
        out_channels,
        kernel_size,
        bias=True,
        dtype=torch.double,
    )

    # same weights
    model.weight = model_torch.weight
    model.bias = model_torch.bias

    outputs = model(inputs)
    outputs_torch = model_torch(inputs)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)



@pytest.mark.order(2)
@pytest.mark.parametrize("out_channels, kernel_size", [(6, 4), (3, 5)])
def test_conv_backward(out_channels, kernel_size):

    B, Cin, H, W = 4, 5, 14, 11

    inputs = torch.randn(
        B, Cin, H, W,
        dtype=torch.double,
        requires_grad=True,
    )

    inputs_torch = inputs.clone().detach().requires_grad_(True)

    model = Conv2d(Cin, out_channels, kernel_size)
    model_torch = torch.nn.Conv2d(
        Cin,
        out_channels,
        kernel_size,
        bias=True,
        dtype=torch.double,
    )

    # same weights
    model.weight = model_torch.weight
    model.bias = model_torch.bias

    # forward
    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)

    # fake loss
    loss = outputs.sum()
    loss_torch = outputs_torch.sum()

    # backward
    loss.backward()
    loss_torch.backward()

    # check grad inputs
    assert torch.allclose(
        inputs.grad,
        inputs_torch.grad,
        atol=1e-6,
    )

    # check grad weights
    assert torch.allclose(
        model.weight.grad,
        model_torch.weight.grad,
        atol=1e-6,
    )

    # check grad bias
    assert torch.allclose(
        model.bias.grad,
        model_torch.bias.grad,
        atol=1e-6,
    )