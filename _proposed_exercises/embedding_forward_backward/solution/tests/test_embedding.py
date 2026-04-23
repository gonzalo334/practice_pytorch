import torch
import pytest

from src.embedding import Embedding


@pytest.mark.order(1)
@pytest.mark.parametrize("shape, padding_idx", [((8,), None), ((4, 5), 2)])
def test_embedding_forward_backward(
    shape: tuple[int, ...], padding_idx: int | None
) -> None:
    torch.manual_seed(0)
    num_embeddings, embedding_dim = 10, 6
    inputs = torch.randint(0, num_embeddings, shape)
    if padding_idx is not None:
        inputs.reshape(-1)[0] = padding_idx

    model = Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=torch.double
    )
    model_torch = torch.nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=torch.double
    )
    model.weight = torch.nn.Parameter(model_torch.weight.detach().clone())

    outputs = model(inputs)
    outputs_torch = model_torch(inputs)
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)

