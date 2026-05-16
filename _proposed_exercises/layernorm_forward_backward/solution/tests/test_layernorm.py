import torch
import pytest

from src.layernorm import LayerNorm


def build_models(
    normalized_shape: tuple[int, ...],
) -> tuple[LayerNorm, torch.nn.LayerNorm]:
    model = LayerNorm(normalized_shape, dtype=torch.double)
    model_torch = torch.nn.LayerNorm(normalized_shape, dtype=torch.double)
    model.weight = torch.nn.Parameter(model_torch.weight.detach().clone())
    model.bias = torch.nn.Parameter(model_torch.bias.detach().clone())
    return model, model_torch


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "shape, normalized_shape",
    [((16, 64), (64,)), ((4, 8, 16, 32), (16, 32))],
)
def test_layernorm_forward(
    shape: tuple[int, ...], normalized_shape: tuple[int, ...]
) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model, model_torch = build_models(normalized_shape)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "shape, normalized_shape",
    [((16, 64), (64,)), ((4, 8, 16, 32), (16, 32))],
)
def test_layernorm_backward(
    shape: tuple[int, ...], normalized_shape: tuple[int, ...]
) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model, model_torch = build_models(normalized_shape)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)
    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)
    assert torch.allclose(model.bias.grad, model_torch.bias.grad, atol=1e-6)
