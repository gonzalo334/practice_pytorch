import torch
import pytest

from src.embedding_bag import EmbeddingBag


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_embedding_bag_forward_backward(mode: str) -> None:
    torch.manual_seed(0)
    num_embeddings, embedding_dim = 11, 5
    inputs = torch.tensor([1, 3, 3, 2, 5, 1, 8, 8, 4])
    offsets = torch.tensor([0, 3, 6])

    model = EmbeddingBag(num_embeddings, embedding_dim, mode=mode, dtype=torch.double)
    model_torch = torch.nn.EmbeddingBag(
        num_embeddings, embedding_dim, mode=mode, dtype=torch.double
    )
    model.weight = torch.nn.Parameter(model_torch.weight.detach().clone())

    outputs = model(inputs, offsets)
    outputs_torch = model_torch(inputs, offsets)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)
