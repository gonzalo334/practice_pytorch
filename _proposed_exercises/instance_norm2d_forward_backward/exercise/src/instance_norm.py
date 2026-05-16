# deep learning libraries
import torch

# other libraries
from typing import Any


class InstanceNorm2dFunction(torch.autograd.Function):
    """
    Class for the implementation of InstanceNorm2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Forward pass of InstanceNorm2d.
        """

        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of InstanceNorm2d.
        """

        # TODO
        raise NotImplementedError


class InstanceNorm2d(torch.nn.Module):
    """
    This is the class that represents InstanceNorm2d.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(num_features, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.empty(num_features, dtype=dtype))
        self.reset_parameters()
        self.fn = InstanceNorm2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.weight, self.bias, self.eps)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
