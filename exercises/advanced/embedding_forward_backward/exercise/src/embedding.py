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

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        """
        This method is the backward of the Embedding layer.
        """

        # TODO


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

