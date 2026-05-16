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

        # TODO
        B, C, L = inputs.shape
        Cout, Cin, K = weight.shape
        ctx.save_for_backward(inputs, weight)
        Lout = L - K + 1
        outputs = torch.zeros((B, Cout, Lout), dtype=inputs.dtype)
        weight = weight.view(1, Cout, 1, Cin*K)
        for li in range(Lout):
            window = inputs[:, :, li:li+K].contiguous().view(B, 1, Cin*K, 1) # B, 1, Cin*K, 1
            outputs[:, :, li] = (weight @ window).squeeze()
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

        # TODO
        inputs, weight = ctx.saved_tensors
        inputs : torch.Tensor
        weight : torch.Tensor
        B, Cin, L = inputs.shape
        Cout, Cin, K = weight.shape
        Lout = L - K + 1
        # B, Cout, Lout = grad_outputs.shape
        grad_weight = torch.zeros_like(weight)
        weight = weight.view(1, Cout, Cin*K)
        grad_inputs = torch.zeros_like(inputs)
        for li in range(Lout):
            window = inputs[:,:, li:li+K].permute(1,2,0).contiguous().view(Cin*K, B).view(Cin*K, 1, B, 1)
            grad_patch = grad_outputs[:, :, li] # B, Cout
            grad_patch = grad_patch.T.view(1, Cout, 1, B)
            grad_weight_folded = (grad_patch @ window).view(Cin, K, Cout).permute(2, 0, 1)
            grad_weight += grad_weight_folded
            grad_inputs[:, :, li:li+K] += (grad_outputs[:, :, li].unsqueeze(1) @ weight).view(B, Cin, K)

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
    
    @property
    def shape(self) -> torch.Size:
        return self.weight.shape