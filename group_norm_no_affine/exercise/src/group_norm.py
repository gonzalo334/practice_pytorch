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

        # TODO

