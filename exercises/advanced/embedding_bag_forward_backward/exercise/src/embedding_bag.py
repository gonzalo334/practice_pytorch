# deep learning libraries
import torch

# other libraries
from typing import Any


class EmbeddingBagFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of EmbeddingBag.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        offsets: torch.Tensor,
        weight: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        Forward pass of EmbeddingBag.
        """

        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, None, torch.Tensor, None]:
        """
        Backward pass of EmbeddingBag.
        """

        # TODO
        raise NotImplementedError


class EmbeddingBag(torch.nn.Module):
    """
    This is the class that represents the EmbeddingBag layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "mean",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        )
        self.mode = mode
        self.reset_parameters()
        self.fn = EmbeddingBagFunction.apply

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, offsets, self.weight, self.mode)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
