# deep learning libraries
import torch

# other libraries
from typing import Any


class AdaptiveAvgPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of AdaptiveAvgPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Forward pass of AdaptiveAvgPool2d.
        """

        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        Backward pass of AdaptiveAvgPool2d.
        """

        # TODO
        raise NotImplementedError


class AdaptiveAvgPool2d(torch.nn.Module):
    """
    This is the class that represents AdaptiveAvgPool2d.
    """

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size
        self.fn = AdaptiveAvgPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.output_size)
