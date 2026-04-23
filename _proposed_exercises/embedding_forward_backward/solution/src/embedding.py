# deep learning libraries
import torch

# other libraries
from typing import Any


class EmbeddingFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of Embedding.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        padding_idx: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the Embedding layer.
        """

        outputs = weight[inputs]
        ctx.save_for_backward(inputs, weight)
        ctx.padding_idx = padding_idx

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        """
        This method is the backward of the Embedding layer.
        """

        inputs, weight = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_outputs_flat = grad_outputs.reshape(-1, weight.shape[1])
        inputs_flat = inputs.reshape(-1)

        if ctx.padding_idx >= 0:
            mask = inputs_flat != ctx.padding_idx
            inputs_flat = inputs_flat[mask]
            grad_outputs_flat = grad_outputs_flat[mask]

        grad_weight.index_add_(0, inputs_flat, grad_outputs_flat)

        return None, grad_weight, None


class Embedding(torch.nn.Module):
    """
    This is the class that represents the Embedding layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        )
        self.padding_idx = -1 if padding_idx is None else padding_idx
        self.reset_parameters()
        self.fn = EmbeddingFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.weight, self.padding_idx)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
        if self.padding_idx >= 0:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

