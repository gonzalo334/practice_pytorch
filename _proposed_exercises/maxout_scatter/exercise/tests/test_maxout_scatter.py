import torch
import pytest

from src.maxout_scatter import Maxout


def maxout_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    z = torch.einsum("bi,koi->bko", inputs, weights) + bias.unsqueeze(0)
    return z.max(dim=1).values


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "batch, input_dim, output_dim, num_units",
    [(4, 5, 3, 2), (6, 4, 7, 5)],
)
def test_maxout_forward_backward(
    batch: int,
    input_dim: int,
    output_dim: int,
    num_units: int,
) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(batch, input_dim, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = Maxout(input_dim, output_dim, num_units, dtype=torch.double)
    weights_torch = model.weights.detach().clone().requires_grad_(True)
    bias_torch = model.bias.detach().clone().requires_grad_(True)

    outputs = model(inputs)
    outputs_torch = maxout_reference(inputs_torch, weights_torch, bias_torch)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(model.weights.grad, weights_torch.grad, atol=1e-6)
    assert torch.allclose(model.bias.grad, bias_torch.grad, atol=1e-6)

