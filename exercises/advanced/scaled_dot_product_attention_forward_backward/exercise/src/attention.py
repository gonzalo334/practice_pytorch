# deep learning libraries
import torch

# other libraries
from typing import Any


class ScaledDotProductAttentionFunction(torch.autograd.Function):
    """
    Class for the implementation of scaled dot-product attention.
    """

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """
        Compute attention.

        Args:
            query: tensor with shape ``[batch, target_length, dim]``.
            key: tensor with shape ``[batch, source_length, dim]``.
            value: tensor with shape ``[batch, source_length, value_dim]``.
            causal: whether future source positions are masked.

        Returns:
            Tensor with shape ``[batch, target_length, value_dim]``.
        """

        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass for attention.
        """

        # TODO
        raise NotImplementedError


class ScaledDotProductAttention(torch.nn.Module):
    """
    Single-head scaled dot-product attention.
    """

    def __init__(self, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.fn = ScaledDotProductAttentionFunction.apply

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return self.fn(query, key, value, self.causal)
