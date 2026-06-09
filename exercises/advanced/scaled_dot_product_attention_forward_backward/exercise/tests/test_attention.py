import math

import torch
import pytest

from src.attention import ScaledDotProductAttention


def attention_reference(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool
) -> torch.Tensor:
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if causal:
        length = scores.shape[-1]
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ value


@pytest.mark.parametrize("causal", [False, True])
def test_attention_forward_backward(causal: bool) -> None:
    torch.manual_seed(0)
    batch, target_length, source_length, dim, value_dim = 3, 5, 5, 4, 6
    query = torch.randn(batch, target_length, dim, dtype=torch.double, requires_grad=True)
    key = torch.randn(batch, source_length, dim, dtype=torch.double, requires_grad=True)
    value = torch.randn(
        batch, source_length, value_dim, dtype=torch.double, requires_grad=True
    )
    query_torch = query.detach().clone().requires_grad_(True)
    key_torch = key.detach().clone().requires_grad_(True)
    value_torch = value.detach().clone().requires_grad_(True)

    model = ScaledDotProductAttention(causal=causal)
    outputs = model(query, key, value)
    outputs_torch = attention_reference(query_torch, key_torch, value_torch, causal)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(query.grad, query_torch.grad, atol=1e-6)
    assert torch.allclose(key.grad, key_torch.grad, atol=1e-6)
    assert torch.allclose(value.grad, value_torch.grad, atol=1e-6)
