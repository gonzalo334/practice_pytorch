import torch
import torch.nn.functional as F

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
