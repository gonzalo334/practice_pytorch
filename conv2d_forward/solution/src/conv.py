"""
This module contains the code for the Conv2d.
"""

# 3pps
import torch


class Conv2d(torch.nn.Module):
    """
    This is the class to implement the Conv2d.

    Attributes:
        weight: weight tensor. Dimensions: [output channels,
            input channels, kernel size, kernel size].
        bias: bias tensor. Dimensions: [output channels].
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        This method is the constructor of Conv2d.

        Args:
            in_channels: input channels to the layer.
            out_channels: output channels to the layer.
            kernel_size: kernel size for the layer.

        Returns:
            None.
        """

        # Call super class
        super().__init__()

        # Set attributes
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Define parameters
        self.weight = torch.nn.Parameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = torch.nn.Parameter(torch.rand(out_channels))

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is thf forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, output channels,
                (height - kernel size + 1), (height - kernel size + 1)].
        """

        # Set outputs sizes
        output_height: int = inputs.shape[2] - self.kernel_size + 1
        output_width: int = inputs.shape[2] - self.kernel_size + 1

        # Define initial tensor
        outputs = torch.empty(
            inputs.shape[0], self.out_channels, output_height, output_width
        ).double()

        # Loops for convolution
        for i in range(output_height):
            for j in range(output_width):
                # Add dims to weight
                weight: torch.Tensor = self.weight.unsqueeze(0)
                inputs_kernel: torch.Tensor = inputs[
                    :, :, i : i + self.kernel_size, j : j + self.kernel_size
                ].unsqueeze(1)

                # Add dims to bias
                bias: torch.Tensor = self.bias.view(1, -1)

                # Compute conv operation
                outputs[:, :, i, j] = (weight * inputs_kernel).sum(dim=(2, 3, 4)) + bias

        return outputs
