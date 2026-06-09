import torch
import pytest

from src.rnn import EmbeddingRNN


def reference_rnn(
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_h: torch.Tensor,
    truncate_steps: int,
) -> torch.Tensor:
    batch, seq_len = token_ids.shape
    hidden_dim = weight_hh.shape[0]
    hidden = torch.zeros(batch, hidden_dim, dtype=embedding_weight.dtype)
    outputs = []
    for time_step in range(seq_len):
        if truncate_steps > 0 and time_step > 0 and time_step % truncate_steps == 0:
            hidden = hidden.detach()
        inputs = embedding_weight[token_ids[:, time_step]]
        hidden = torch.tanh(inputs @ weight_ih.T + hidden @ weight_hh.T + bias_h)
        outputs.append(hidden)
    return torch.stack(outputs, dim=1)


@pytest.mark.order(1)
@pytest.mark.parametrize("truncate_steps", [0, 2])
def test_embedding_rnn_forward_backward(truncate_steps: int) -> None:
    torch.manual_seed(0)
    token_ids = torch.randint(0, 7, (3, 5))

    model = EmbeddingRNN(7, 4, 6, truncate_steps=truncate_steps, dtype=torch.double)
    params = [
        model.embedding_weight.detach().clone().requires_grad_(True),
        model.weight_ih.detach().clone().requires_grad_(True),
        model.weight_hh.detach().clone().requires_grad_(True),
        model.bias_h.detach().clone().requires_grad_(True),
    ]

    outputs = model(token_ids)
    outputs_torch = reference_rnn(token_ids, *params, truncate_steps)
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(model.embedding_weight.grad, params[0].grad, atol=1e-6)
    assert torch.allclose(model.weight_ih.grad, params[1].grad, atol=1e-6)
    assert torch.allclose(model.weight_hh.grad, params[2].grad, atol=1e-6)
    assert torch.allclose(model.bias_h.grad, params[3].grad, atol=1e-6)

