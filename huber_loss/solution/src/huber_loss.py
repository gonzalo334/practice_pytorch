"""
This module contains the code for HuberLoss.
"""

# Standard libraries
from typing import Any

# 3pps
import torch


class HuberLossFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the HuberLoss.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, targets: torch.Tensor, delta: float
    ) -> torch.Tensor:
        """
        This is the forward method of the HuberLoss.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # Compute mask
        mask = torch.zeros_like(inputs).bool()
        mask[torch.abs(inputs - targets) < delta] = True

        # Init outputs
        outputs: torch.Tensor = torch.zeros_like(inputs)

        # Compute outputs
        outputs[mask] = 0.5 * (inputs[mask] - targets[mask]) ** 2
        outputs[~mask] = delta * (
            torch.abs(inputs[~mask] - targets[~mask]) - 0.5 * delta
        )

        # Save elements for the backward
        ctx.delta = delta
        ctx.save_for_backward(mask, inputs, targets)

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        This method is the backward of the HuberLoss.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
            None.
            None.
        """

        # load tensors from the forward
        mask, inputs, targets = ctx.saved_tensors

        # Compute gradients inside condition
        grad_input: torch.Tensor = torch.ones_like(inputs)
        grad_input[mask] = inputs[mask] - targets[mask]

        # Compute gradients outside condition
        signed_mask = torch.empty_like(inputs)
        signed_mask[(inputs - targets) < 0] = -1
        signed_mask[(inputs - targets) > 0] = 1
        grad_input[~mask] = ctx.delta * signed_mask[~mask]

        # Add to grad output
        grad_input *= grad_output

        return grad_input, None, None


class HuberLoss(torch.nn.Module):
    """
    This is the class that represents the HuberLoss Layer.
    """

    def __init__(self, delta: float = 1.0):
        """
        This method is the constructor of the HuberLoss layer.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.delta = delta

        # Set function
        self.fn = HuberLossFunction.apply

        return None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs, targets, self.delta)
