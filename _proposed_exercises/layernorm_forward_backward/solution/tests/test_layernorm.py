import torch
import pytest

from src.layernorm import LayerNorm


@pytest.mark.order(1)
@pytest.mark.parametrize("shape, normalized_shape", [((4, 5), (5,)), ((2, 3, 4), (3, 4))])
def test_layernorm_forward_backward(
    shape: tuple[int, ...], normalized_shape: tuple[int, ...]
) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = LayerNorm(normalized_shape, dtype=torch.double)
    model_torch = torch.nn.LayerNorm(normalized_shape, dtype=torch.double)
    model.weight = torch.nn.Parameter(model_torch.weight.detach().clone())
    model.bias = torch.nn.Parameter(model_torch.bias.detach().clone())

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)
    assert torch.allclose(model.bias.grad, model_torch.bias.grad, atol=1e-6)

