import torch
import pytest

from src.convtranspose import ConvTranspose2d


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "stride, padding, output_padding, groups, dilation",
    [((1, 1), (0, 0), (0, 0), 1, (1, 1)), ((2, 1), (1, 0), (1, 0), 2, (1, 1))],
)
def test_convtranspose2d_forward_backward(
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    groups: int,
    dilation: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    batch, in_channels = 2, 4
    out_channels = 6 if groups == 2 else 5
    kernel_size = (3, 2)

    inputs = torch.randn(batch, in_channels, 5, 4, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        dtype=torch.double,
    )
    model_torch = torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
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

