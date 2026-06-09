# deep learning libraries
import torch

# other libraries
import math
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

        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = query @ key.transpose(-2, -1) * scale
        if causal:
            target_length, source_length = scores.shape[-2:]
            mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=scores.device,
            ).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        outputs = weights @ value

        ctx.save_for_backward(query, key, value, weights)
        ctx.scale = scale
        ctx.causal = causal

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass for attention.
        """

        query, key, value, weights = ctx.saved_tensors
        scale: float = ctx.scale

        grad_value = weights.transpose(-2, -1) @ grad_outputs
        grad_weights = grad_outputs @ value.transpose(-2, -1)
        grad_scores = weights * (
            grad_weights - (grad_weights * weights).sum(dim=-1, keepdim=True)
        )

        grad_query = grad_scores @ key * scale
        grad_key = grad_scores.transpose(-2, -1) @ query * scale

        return grad_query, grad_key, grad_value, None


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
