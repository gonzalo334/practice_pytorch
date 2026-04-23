"""
This module contains the code to implement the GroupNorm.
"""

# 3pps
import torch


class GroupNorm(torch.nn.Module):
    """
    This class implements the GroupNorm layer of torch.

    Attributes:
        num_groups: Number of groups.
        num_channels: Number of channels.
        eps: epsilon to avoid division by 0.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        """
        This method is the constructor of GroupNorm class.

        Args:
            num_groups: Number of groups.
            num_channels: Number of channels.
            eps: epsilon to avoid division by 0. Defaults to 1e-5.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height,
                width].
        """

        # Reshape inputs
        inputs = inputs.view(
            inputs.shape[0],
            self.num_groups,
            self.num_channels // self.num_groups,
            inputs.shape[2],
            inputs.shape[3],
        )

        # Compute inputs and var
        mean: torch.Tensor = torch.mean(inputs, dim=(2, 3, 4), keepdim=True)
        var: torch.Tensor = torch.var(
            inputs, dim=(2, 3, 4), keepdim=True, unbiased=False
        )

        # Compute outputs
        outputs: torch.Tensor = (inputs - mean) / torch.sqrt(var + self.eps)

        # Reshape outputs
        outputs = outputs.view(
            outputs.shape[0],
            self.num_channels,
            outputs.shape[3],
            outputs.shape[4],
        )

        return outputs
