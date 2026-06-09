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
        B, Cin, L = inputs.shape
        Cout, Cin, K = weight.shape
        Lout = L - K + 1
        outputs = torch.zeros(B, Cout, Lout)
        inputs_f = inputs.unsqueeze(-1) # B, Cin, L, 1
        inputs_unfolded = torch.nn.functional.unfold(inputs_f, (K, 1)) # B, Cin*K, Lout
        weight_unfolded = weight.view(Cout, Cin*K).unsqueeze(0) # 1, Cout, Cin*K
        outputs = weight_unfolded @ inputs_unfolded # B, Cout, Lout
        ctx.save_for_backward(inputs, weight, outputs)
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
        inputs, weight, outputs = ctx.saved_tensors
        B, Cin, L = inputs.shape
        Cout, Cin, K = weight.shape
        Lout = L - K + 1
        # B, Cout, Lout = grad_outputs.shape

        inputs_f = inputs.unsqueeze(-1) # B, Cin, L, 1
        inputs_unfolded = torch.nn.functional.unfold(inputs_f, (K, 1)) # B, Cin*K, Lout
        grad_weights_batches = grad_outputs @ inputs_unfolded.transpose(1,2) # B, Cout, Cin*K
        grad_weights = grad_weights_batches.view(B, Cout, Cin, K).sum(dim=0) # Cout, Cin, K
        
        weights_unfolded = weight.view(1, Cout, Cin*K).transpose(1,2) # 1, Cin*K, Cout
        grad_inputs_unfolded = weights_unfolded @ grad_outputs # B, Cin*K, Lout

        # grad_inputs = grad_inputs.view(B, Cin, L) # B, Cin, L
        # El view funcionaría si no solaparan ventanas
        grad_inputs = torch.nn.functional.fold(grad_inputs_unfolded, (L, 1), (K, 1)) # B, Cin, L, 1
        return grad_inputs.squeeze(-1), grad_weights
        

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
