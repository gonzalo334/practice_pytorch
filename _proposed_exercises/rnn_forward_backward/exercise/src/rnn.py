# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class EmbeddingRNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of a tanh RNN
    with embedding inputs.
    """

    @staticmethod
    def forward(
        ctx: Any,
        token_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_h: torch.Tensor,
        truncate_steps: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the RNN.
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        This method is the backward of the RNN.
        """

        # TODO


class EmbeddingRNN(torch.nn.Module):
    """
    This is the class that represents the tanh RNN with embedding inputs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        truncate_steps: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.truncate_steps = truncate_steps
        self.embedding_weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        )
        self.weight_ih = torch.nn.Parameter(
            torch.empty(hidden_dim, embedding_dim, dtype=dtype)
        )
        self.weight_hh = torch.nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=dtype)
        )
        self.bias_h = torch.nn.Parameter(torch.empty(hidden_dim, dtype=dtype))
        self.reset_parameters()
        self.fn = EmbeddingRNNFunction.apply

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.fn(
            token_ids,
            self.embedding_weight,
            self.weight_ih,
            self.weight_hh,
            self.bias_h,
            self.truncate_steps,
        )

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.embedding_weight, mean=0.0, std=0.1)
        bound = 1 / math.sqrt(self.weight_ih.shape[1])
        torch.nn.init.uniform_(self.weight_ih, -bound, bound)
        torch.nn.init.uniform_(self.weight_hh, -bound, bound)
        torch.nn.init.uniform_(self.bias_h, -bound, bound)

