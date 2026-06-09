import torch
import pytest

from src.instance_norm import InstanceNorm2d


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (3, 4, 6, 3)])
def test_instance_norm2d_forward_backward(shape: tuple[int, int, int, int]) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = InstanceNorm2d(shape[1], dtype=torch.double)
    model_torch = torch.nn.InstanceNorm2d(
        shape[1], affine=True, track_running_stats=False, dtype=torch.double
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
