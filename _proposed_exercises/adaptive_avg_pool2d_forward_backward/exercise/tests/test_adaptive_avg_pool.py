import torch
import pytest

from src.adaptive_avg_pool import AdaptiveAvgPool2d


@pytest.mark.parametrize("output_size", [(2, 3), (4, 2)])
def test_adaptive_avg_pool2d_forward_backward(
    output_size: tuple[int, int]
) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 3, 5, 7, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = AdaptiveAvgPool2d(output_size)
    model_torch = torch.nn.AdaptiveAvgPool2d(output_size)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
