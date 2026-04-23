# deep learning libraries
import torch

# other libraries
from typing import Any


class LayerNormFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of LayerNorm.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        normalized_shape: tuple[int, ...],
        eps: float,
    ) -> torch.Tensor:
        """
        This is the forward method of LayerNorm.
        """

        dims = tuple(range(inputs.dim() - len(normalized_shape), inputs.dim()))
        mean = inputs.mean(dim=dims, keepdim=True)
        var = inputs.var(dim=dims, unbiased=False, keepdim=True)
        inv_std = torch.rsqrt(var + eps)
        normalized = (inputs - mean) * inv_std
        outputs = normalized * weight + bias

        ctx.save_for_backward(normalized, inv_std, weight)
        ctx.normalized_shape = normalized_shape
        ctx.input_dim = inputs.dim()

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This method is the backward of LayerNorm.
        """

        normalized, inv_std, weight = ctx.saved_tensors
        normalized_shape: tuple[int, ...] = ctx.normalized_shape
        dims = tuple(range(ctx.input_dim - len(normalized_shape), ctx.input_dim))
        leading_dims = tuple(range(ctx.input_dim - len(normalized_shape)))
        elements = 1
        for value in normalized_shape:
            elements *= value

        grad_normalized = grad_outputs * weight
        grad_sum = grad_normalized.sum(dim=dims, keepdim=True)
        grad_normalized_sum = (grad_normalized * normalized).sum(dim=dims, keepdim=True)
        grad_inputs = (
            grad_normalized
            - grad_sum / elements
            - normalized * grad_normalized_sum / elements
        ) * inv_std

        grad_weight = (grad_outputs * normalized).sum(dim=leading_dims)
        grad_bias = grad_outputs.sum(dim=leading_dims)

        return grad_inputs, grad_weight, grad_bias, None, None


class LayerNorm(torch.nn.Module):
    """
    This is the class that represents the LayerNorm layer.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.empty(*normalized_shape, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.empty(*normalized_shape, dtype=dtype))

        self.reset_parameters()
        self.fn = LayerNormFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.weight, self.bias, self.normalized_shape, self.eps)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

