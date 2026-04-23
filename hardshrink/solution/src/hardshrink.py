"""
This module contains the code for Hardshrink.
"""

# Standard libraries
from typing import Any

# 3pps
import torch


class HardshrinkFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Hardshrink.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambd: float) -> torch.Tensor:
        """
        This is the forward method of the Hardshrink.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # compute forward
        outputs = inputs.clone()
        mask: torch.Tensor = (outputs >= -lambd) & (outputs <= lambd)
        outputs[mask] = 0

        # save tensors for the backward
        ctx.save_for_backward(mask)

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        This method is the backward of the Hardshrink.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
            None.
        """

        # load tensors from the forward
        (mask,) = ctx.saved_tensors

        # compute gradients
        grad_input: torch.Tensor = torch.ones_like(mask).float()
        grad_input[mask] = 0
        grad_input *= grad_output

        return grad_input, None


class Hardshrink(torch.nn.Module):
    """
    This is the class that represents the Hardshrink Layer.
    """

    def __init__(self, lambd: float = 0.5):
        """
        This method is the constructor of the Hardshrink layer.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.lambd = lambd

        # Set function
        self.fn = HardshrinkFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs, self.lambd)
