import torch
import torch.nn.functional as F
import pytest

from src.cross_entropy import CrossEntropyLoss


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_forward_backward(reduction: str) -> None:
    torch.manual_seed(0)
    logits = torch.randn(7, 5, dtype=torch.double, requires_grad=True)
    targets = torch.tensor([0, 4, 2, -1, 3, 1, -1])
    logits_torch = logits.detach().clone().requires_grad_(True)

    loss_fn = CrossEntropyLoss(ignore_index=-1, reduction=reduction)
    outputs = loss_fn(logits, targets)
    outputs_torch = F.cross_entropy(
        logits_torch, targets, ignore_index=-1, reduction=reduction
    )

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(logits.grad, logits_torch.grad, atol=1e-6)
