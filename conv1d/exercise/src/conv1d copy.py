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
        B, Cin, Lin = inputs.shape
        Cout, Cin, K = weight.shape
        ctx.save_for_backward(inputs, weight)
        outputs = torch.empty((B, Cout, Lin-K+1), dtype=inputs.dtype)#torch.double)
        weight = weight.unsqueeze(0) # 1, Cout, Cin, K
        Lout = Lin - K + 1
        for i in range(Lout):
            patch = inputs[:,:, i:i+K] # B, Cin, K
            patch = patch.unsqueeze(1) # B, 1, Cin, K
            multiplied = (patch * weight).sum(dim=(2, 3))
            # En las matrices cuando multiplicamos lo hacemos sobre la dimension Cin*K, que corresponden a la 2 y la 3
            outputs[:,:, i] = multiplied
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
        B, Cin, Lin = inputs.shape
        Cout, Cin, K = weight.shape
        B, Cout, Lout = grad_outputs.shape

        grad_inputs = torch.zeros_like(inputs)
        grad_weight = torch.zeros_like(weight)

        for i in range(Lout):
            patch_grad_out = grad_outputs[:,:, i] # B, Cout
            patch_grad_out = patch_grad_out.unsqueeze(-1).unsqueeze(-1) # B, Cout, 1, 1
            grad_inputs[:,:,i:i+K] += (patch_grad_out * weight).sum(dim=1)
            patch_inputs = inputs[:,:, i:i+K]
            grad_weight = (patch_inputs.unsqueeze(1) * patch_grad_out).sum(dim=0)

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
    
    @property
    def shape(self) -> torch.Size:
        return self.weight.shape
    
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
