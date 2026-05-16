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

        Args:
            ctx: context used to save tensors for the backward pass.
            inputs: input tensor with shape ``[*B, *normalized_shape]``.
            weight: learnable scale with shape ``normalized_shape``.
            bias: learnable offset with shape ``normalized_shape``.
            normalized_shape: last dimensions to normalize.
            eps: small constant for numerical stability.

        Returns:
            Output tensor with the same shape as ``inputs``.
        """

        # TODO
        normalized_dims = len(normalized_shape)
        denominator = torch.sqrt(torch.var(inputs, dim=torch.arange(-normalized_dims,0).tolist(), unbiased=False, keepdim=True) + eps) #
        inputs_mean = torch.mean(inputs, dim=torch.arange(-normalized_dims,0).tolist(), keepdim=True)
        numerator = inputs - inputs_mean
        outputs = numerator * weight / denominator + bias
        ctx.normalized_shape = normalized_shape
        ctx.save_for_backward(numerator, denominator, inputs, weight)
        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This method is the backward of LayerNorm.

        Args:
            ctx: context containing tensors saved during ``forward``.
            grad_outputs: upstream gradients with the same shape as ``inputs``.

        Returns:
            Gradients for ``inputs``, ``weight``, and ``bias``.
        """

        # TODO
        numerator, denominator, inputs, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        normalized_dims = len(normalized_shape)

        batch_dims = tuple(range(0, inputs.dim() - normalized_dims))
        norm_dims = tuple(range(-normalized_dims, 0))

        x_hat = numerator / denominator

        grad_weights = (x_hat * grad_outputs).sum(dim=batch_dims)
        grad_bias = grad_outputs.sum(dim=batch_dims)

        g = grad_outputs * weight

        grad_inputs = (
            g
            - g.mean(dim=norm_dims, keepdim=True)
            - x_hat * (g * x_hat).mean(dim=norm_dims, keepdim=True)
        ) / denominator

        return grad_inputs, grad_weights, grad_bias, None, None

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
        """
        Compute the LayerNorm output.

        Args:
            inputs: input tensor with shape ``[*B, *normalized_shape]``.

        Returns:
            Output tensor with the same shape as ``inputs``.
        """

        return self.fn(inputs, self.weight, self.bias, self.normalized_shape, self.eps)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
