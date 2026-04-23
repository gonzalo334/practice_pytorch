# deep learning libraries
import torch

# other libraries
import pytest
from typing import Optional

# own modules
from src.embedding import Embedding
from src.utils import set_seed


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "shape, num_embeddings, embedding_dim, padding_idx",
    [((64,), 20, 32, None), ((128,), 64, 16, 2)],
)
def test_embedding(
    shape: tuple[int, ...],
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int],
) -> None:
    """
    This function is the test for the backward of the embeddings.

    Args:
        shape: shape for the inputs.
        num_embeddings: number of embeddings.
        embedding_dim: embeddings dimension.
        padding_idx: padding index.
    """

    for seed in range(10):
        # set seed and define inputs
        set_seed(seed)
        inputs: torch.Tensor = torch.randint(high=num_embeddings, size=shape)

        # define models
        set_seed(seed)
        model: torch.nn.Module = Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        set_seed(seed)
        model_torch: torch.nn.Module = torch.nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

        # compute backward
        outputs = model(inputs)
        model.zero_grad()
        outputs.sum().backward()
        if model.weight.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_weight: torch.Tensor = model.weight.grad.clone()

        # compute backward of pytorch model
        outputs_torch = model_torch(inputs)
        model_torch.zero_grad()
        outputs_torch.sum().backward()
        if model_torch.weight.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_weight_torch: torch.Tensor = model_torch.weight.grad.clone()

        # check grad weight size
        assert grad_weight.shape == grad_weight_torch.shape, (
            f"Incorrect grad inputs shape, expected {grad_weight_torch.shape}, got "
            f"{grad_weight.shape}"
        )

        # check grad weight
        assert torch.allclose(
            grad_weight, grad_weight_torch, atol=1e-10
        ), "Incorrect grad inputs"

    return None
