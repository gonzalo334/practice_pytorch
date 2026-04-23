# deep learning libraries
import torch

# other libraries
from typing import Any, Optional


class SoftshrinkFuncion(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Softshrink.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        lambd: float,
    ) -> torch.Tensor:
        """
        This is the forward method of the Softshrink.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, *].

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        This method is the backward of the Softshrink.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [batch, *].

        Returns:
            inputs gradients. Dimensions: [batch, *].
        """

        # TODO


class Softshrink(torch.nn.Module):
    """
    This is the class that represents the Softshrink Layer.
    """

    padding_idx: int

    def __init__(self, lambd: float = 0.5) -> None:
        """
        This method is the constructor of the Softshrink layer.
        """

        # call super class constructor
        super().__init__()

        # init parameters corectly
        self.lambd = lambd

        self.fn = SoftshrinkFuncion.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, *].

        Returns:
            outputs tensor. Dimensions: [batch, *].
        """

        return self.fn(inputs, self.lambd)
