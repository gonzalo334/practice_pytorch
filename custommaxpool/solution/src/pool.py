"""
This module contains the code to implement CustomMaxPool2d.
"""

# Standard libraries
from typing import Any

# 3pps
import torch
import torch.nn.functional as F


class CustomMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, kernel_size: int, num_groups: int
    ) -> torch.Tensor:
        """
        This is the forward method of the CustomMaxPool2d.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, number of groups,
                height - kernel size + 1,
                width - kernel size + 1].
        """

        # Compute output size
        output_size: int = inputs.shape[2] - kernel_size + 1

        # Compute number of channels per groups
        num_channels_group: int = inputs.shape[1] // num_groups

        # Transform dimensions
        outputs: torch.Tensor = inputs.clone().view(
            -1, num_channels_group, *inputs.shape[2:]
        )

        # Apply unfold
        outputs = F.unfold(outputs, kernel_size=kernel_size)

        # Get the maximum
        outputs, indexes = torch.max(outputs, dim=1, keepdim=True)

        # Fold
        outputs = F.fold(outputs, output_size=output_size, kernel_size=1)
        outputs = outputs.view(-1, num_groups, output_size, output_size)

        # Save for backward
        ctx.save_for_backward(
            inputs, indexes, torch.tensor(kernel_size), torch.tensor(num_channels_group)
        )

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        This method implements the backward pass of the layer.

        Args:
            grad_outputs: Outputs gradients. Dimensions: [batch size,
                number of groups, height - kernel size + 1,
                width - kernel size + 1].

        Returns:
            Gradients of the inputs. Dimensions: [batch size,
                number of channels, height, width].
            None.
            None.
        """

        # Get tensors from forward
        inputs, indexes, kernel_size, num_channels_group = ctx.saved_tensors

        # Define grad inputs
        grad_inputs: torch.Tensor = torch.zeros_like(inputs)
        grad_inputs = grad_inputs.view(
            -1, num_channels_group.item(), *grad_inputs.shape[2:]
        )

        # Unfold
        grad_inputs = F.unfold(grad_inputs, kernel_size=kernel_size.item())

        # Assign grads based on indexes
        mask: torch.Tensor = torch.permute(
            F.one_hot(indexes, num_classes=grad_inputs.shape[1]).squeeze(1), (0, 2, 1)
        )
        grad_inputs[mask == 1] = 1

        # Add grad outputs
        grad_inputs = grad_inputs * F.unfold(grad_outputs, kernel_size=1).view(
            grad_inputs.shape[0], 1, -1
        )

        # Fold
        grad_inputs = F.fold(
            grad_inputs, output_size=inputs.shape[2], kernel_size=kernel_size.item()
        )
        grad_inputs = grad_inputs.view(*inputs.shape)

        return grad_inputs, None, None


class CustomMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size: int, num_groups: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            kernel_size: Kernel size.
            num_groups: Number of groups.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        # Set function
        self.fn = CustomMaxPool2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, ].
        """

        # Get outputs
        outputs: torch.Tensor = self.fn(inputs, self.kernel_size, self.num_groups)

        return outputs
