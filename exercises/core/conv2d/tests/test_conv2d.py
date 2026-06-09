import torch
import torch.nn.functional as F
import pytest

from src.conv2d import Conv2dFunction


def _run_custom_conv(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    padding: int,
    stride: int,
) -> torch.Tensor:
    return Conv2dFunction.apply(inputs, weight, bias, padding, stride)


def test_forward_matches_pytorch_basic_case():
    inputs = torch.tensor(
        [[[[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0],
           [7.0, 8.0, 9.0]]]]
    )
    weight = torch.tensor([[[[1.0, 0.0],
                             [0.0, -1.0]]]])
    bias = torch.tensor([0.5])

    custom = _run_custom_conv(inputs, weight, bias, padding=0, stride=1)
    expected = F.conv2d(inputs, weight, bias, padding=0, stride=1)

    assert custom.shape == expected.shape
    assert torch.allclose(custom, expected, atol=1e-6)


def test_forward_matches_pytorch_with_padding_and_stride():
    torch.manual_seed(0)
    inputs = torch.randn(2, 3, 6, 6)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)

    custom = _run_custom_conv(inputs, weight, bias, padding=1, stride=2)
    expected = F.conv2d(inputs, weight, bias, padding=1, stride=2)

    assert custom.shape == expected.shape
    assert torch.allclose(custom, expected, atol=1e-5)


@pytest.mark.parametrize(
    "batch,in_channels,out_channels,height,width,kernel_size,padding,stride",
    [
        (3, 4, 5, 17, 19, 5, 2, 2),
        (4, 6, 7, 20, 18, 3, 1, 1),
        (2, 5, 8, 23, 21, 4, 0, 3),
    ],
)
def test_forward_matches_pytorch_larger_inputs_and_values(
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    padding: int,
    stride: int,
):
    inputs = torch.linspace(
        -75.0,
        75.0,
        steps=batch * in_channels * height * width,
        dtype=torch.double,
    ).view(batch, in_channels, height, width)
    weight = torch.linspace(
        -12.0,
        12.0,
        steps=out_channels * in_channels * kernel_size * kernel_size,
        dtype=torch.double,
    ).view(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.linspace(-30.0, 30.0, steps=out_channels, dtype=torch.double)

    custom = _run_custom_conv(inputs, weight, bias, padding=padding, stride=stride)
    expected = F.conv2d(inputs, weight, bias, padding=padding, stride=stride)

    assert custom.shape == expected.shape
    assert torch.allclose(custom, expected, rtol=1e-10, atol=1e-8)


def test_backward_matches_pytorch_gradients():
    torch.manual_seed(1)

    inputs_custom = torch.randn(2, 2, 5, 5, dtype=torch.double, requires_grad=True)
    weight_custom = torch.randn(3, 2, 3, 3, dtype=torch.double, requires_grad=True)
    bias_custom = torch.randn(3, dtype=torch.double, requires_grad=True)

    inputs_ref = inputs_custom.detach().clone().requires_grad_(True)
    weight_ref = weight_custom.detach().clone().requires_grad_(True)
    bias_ref = bias_custom.detach().clone().requires_grad_(True)

    custom_output = _run_custom_conv(
        inputs_custom, weight_custom, bias_custom, padding=1, stride=1
    )
    ref_output = F.conv2d(inputs_ref, weight_ref, bias_ref, padding=1, stride=1)

    grad_output = torch.randn_like(ref_output)

    custom_output.backward(grad_output)
    ref_output.backward(grad_output)

    assert torch.allclose(inputs_custom.grad, inputs_ref.grad, atol=1e-6)
    assert torch.allclose(weight_custom.grad, weight_ref.grad, atol=1e-6)
    assert torch.allclose(bias_custom.grad, bias_ref.grad, atol=1e-6)


@pytest.mark.parametrize(
    "padding,stride",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
def test_backward_matches_pytorch_larger_gradients(padding: int, stride: int):
    torch.manual_seed(10 + padding + stride)

    inputs_custom = (torch.randn(3, 4, 13, 15, dtype=torch.double) * 20).requires_grad_()
    weight_custom = (torch.randn(5, 4, 3, 3, dtype=torch.double) * 7).requires_grad_()
    bias_custom = (torch.randn(5, dtype=torch.double) * 11).requires_grad_()

    inputs_ref = inputs_custom.detach().clone().requires_grad_(True)
    weight_ref = weight_custom.detach().clone().requires_grad_(True)
    bias_ref = bias_custom.detach().clone().requires_grad_(True)

    custom_output = _run_custom_conv(
        inputs_custom, weight_custom, bias_custom, padding=padding, stride=stride
    )
    ref_output = F.conv2d(inputs_ref, weight_ref, bias_ref, padding=padding, stride=stride)

    grad_output = torch.randn_like(ref_output) * 5

    custom_output.backward(grad_output)
    ref_output.backward(grad_output)

    assert custom_output.shape == ref_output.shape
    assert torch.allclose(inputs_custom.grad, inputs_ref.grad, rtol=1e-9, atol=1e-7)
    assert torch.allclose(weight_custom.grad, weight_ref.grad, rtol=1e-9, atol=1e-7)
    assert torch.allclose(bias_custom.grad, bias_ref.grad, rtol=1e-9, atol=1e-7)
