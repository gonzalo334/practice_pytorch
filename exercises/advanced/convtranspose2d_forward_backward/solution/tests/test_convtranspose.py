import torch
import pytest

from src.convtranspose import ConvTranspose2d


CASES = [
    ((6, 7), (5, 6), (5, 5), (5, 6)),
    ((7, 6), (6, 5), (5, 5), (6, 5)),
]


def _make_models(
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[ConvTranspose2d, torch.nn.ConvTranspose2d]:
    in_channels = 6
    out_channels = 7
    kernel_size = (5, 6)

    model = ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
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
        dilation=dilation,
        dtype=torch.double,
    )
    model.weight = torch.nn.Parameter(model_torch.weight.detach().clone())
    model.bias = torch.nn.Parameter(model_torch.bias.detach().clone())

    return model, model_torch


@pytest.mark.order(5)
@pytest.mark.parametrize(
    "stride, padding, output_padding, dilation",
    CASES,
)
def test_convtranspose2d_forward(
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    torch.manual_seed(5)
    batch = 5
    in_channels = 6

    inputs = torch.randn(batch, in_channels, 8, 9, dtype=torch.double)

    model, model_torch = _make_models(stride, padding, output_padding, dilation)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=5e-6)


@pytest.mark.order(6)
@pytest.mark.parametrize(
    "stride, padding, output_padding, dilation",
    CASES,
)
def test_convtranspose2d_backward(
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> None:
    torch.manual_seed(5)
    batch = 5
    in_channels = 6

    inputs = torch.randn(batch, in_channels, 8, 9, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model, model_torch = _make_models(stride, padding, output_padding, dilation)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=5e-6)
    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=5e-6)
    assert torch.allclose(model.bias.grad, model_torch.bias.grad, atol=5e-6)
