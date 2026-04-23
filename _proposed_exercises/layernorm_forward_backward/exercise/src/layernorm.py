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

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This method is the backward of LayerNorm.
        """

        # TODO


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

