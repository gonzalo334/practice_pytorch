import torch
import pytest

from src.grouped_conv import GroupedConv2d


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "groups, stride, padding, dilation",
    [(2, (1, 1), (1, 1), (1, 1)), (3, (2, 1), (0, 1), (1, 1))],
)
def test_grouped_conv2d_forward_backward(
    groups: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    in_channels = 6
    out_channels = 12
    kernel_size = (3, 2)

    inputs = torch.randn(2, in_channels, 8, 7, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = GroupedConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=torch.double,
    )
    model_torch = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=torch.double,
    )
    model.weight = torch.nn.Parameter(model_torch.weight.detach().clone())
    model.bias = torch.nn.Parameter(model_torch.bias.detach().clone())

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)
    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)
    assert torch.allclose(model.bias.grad, model_torch.bias.grad, atol=1e-6)

