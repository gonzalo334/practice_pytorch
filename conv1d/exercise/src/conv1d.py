"""
This module contains the code for the Conv1d implementation.
"""

# Standard libraries
from typing import Any

# 3pps
import torch


class Conv1dFunction(torch.autograd.Function):
    """
    This class implements the Conv1d with the autograd.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            ctx: Context to save variables.
            inputs: Inputs tensor. Dimensions: [batch size,
                number of input channels, sequence length].
            weight: Weight tensor. Dimensions:

        Returns:
            Outputs tensor. Dimensions: [batch size,
                number of output channels,
                sequence length - kernel size + 1].
        """

        # Get output size and define outputs
        output_size: int = inputs.shape[2] - weight.shape[2] + 1
        outputs: torch.Tensor = torch.zeros(
            (inputs.shape[0], weight.shape[0], output_size), dtype=torch.double
        )

        # Iter over spatial dimension
        for i in range(output_size):
            outputs[:, :, i] = (
                inputs[:, :, i : i + weight.shape[2]].unsqueeze(1) * weight.unsqueeze(0)
            ).sum(dim=(2, 3))

        # Save for backward
        ctx.save_for_backward(inputs, weight)

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method is the backward pass of the layer.

        Args:
            ctx: Context to save variables.
            grad_outputs: Gradients of the outputs. Dimensions:
                [batch size, number output channels,
                sequence length - kernel size + 1].

        Returns:
            Gradients of the inputs. Dimensions: [batch size,
                number input channels, sequence length].
            Gradients of the weight. Dimensions:
                [number of output channels, number of input channels,
                kernel size].
        """

        # Get tensors from forward
        inputs, weight = ctx.saved_tensors

        # Init grad inputs
        grad_inputs = torch.zeros_like(inputs)

        # Iter over spatial dimension
        for i in range(grad_outputs.shape[2]):
            grad_inputs[:, :, i : i + weight.shape[2]] += (
                grad_outputs[:, :, i].unsqueeze(1).unsqueeze(-1)
                * weight.unsqueeze(0).permute(0, 2, 1, 3)
            ).sum(dim=2)

        grad_weight: torch.Tensor = (
            grad_inputs.unsqueeze(1).unsqueeze(-2) * weight.unsqueeze(0).unsqueeze(-1)
        ).sum(dim=(0, -1))

        return grad_inputs, grad_weight


class Conv1d(torch.nn.Module):
    """
    This class implements the Conv1d without bias.

    Attributes:
        weight: Weight tensor. Dimensions: [number of output channels,
            number of input channels, kernel size].
        fn: Autograd function to implement forward and backward.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size for the convolution.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.weight = torch.nn.Parameter(
            torch.rand((out_channels, in_channels, kernel_size), dtype=torch.double)
        )
        self.shape = self.weight.shape

        # Set function
        self.fn = Conv1dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the class.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size,
                number of input channels, sequence length].

        Returns:
            Outputs tensor. Dimensions: [batch size,
                number of output channels,
                sequence length - kernel size + 1].
        """

        # Compute outputs
        outputs: torch.Tensor = self.fn(inputs, self.weight)

        return outputs
