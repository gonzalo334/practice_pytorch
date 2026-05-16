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

        # TODO
        B, Cin, H, W = inputs.shape
        G, K = num_groups, kernel_size
        Hout, Wout = H - K + 1, W - K + 1
        outputs = torch.zeros((B, num_groups, Hout, Wout), dtype=inputs.dtype)

        inputs_unfolded = torch.nn.functional.unfold(inputs, K) # B, Cin*K*K, Hout*Wout
        inputs_windows = inputs_unfolded.view(B, Cin, K*K, -1) # B, Cin, K*K, Hout*Wout
        inputs_grouped = inputs_windows.view(B, G, (Cin // G) * K*K, -1) # B, G, (Cin // G) * K, Hout*Wout
        values, indexes = torch.max(inputs_grouped, dim=2) 
        outputs = values.view(B, G, Hout, Wout)

        ctx.G = num_groups 
        ctx.K = K
        ctx.save_for_backward(inputs, outputs, indexes)
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

        # TODO
        inputs, outputs, indexes = ctx.saved_tensors
        K, G = ctx.K, ctx.G
        B, Cin, H, W = inputs.shape
        Hout, Wout = H - K + 1, W - K + 1

        inputs_max_one_hot = torch.nn.functional.one_hot(indexes)   # B, G, Hout*Wout, (Cin // G) * K*K
        grad_outputs_unfolded = grad_outputs.view(B, G, Hout*Wout, 1) # B, G, Hout*Wout, 1
        grad_inputs_multiplied = grad_outputs_unfolded * inputs_max_one_hot # B, G, Hout*Wout, (Cin // G)*K*K
        grad_inputs_transposed = grad_inputs_multiplied.view(B, G, (Cin // G)*K*K, Hout*Wout) # B, G, (Cin // G) * K*K, Hout*Wout
        grad_inputs_transposed = grad_inputs_multiplied.transpose(2,3) # B, G, (Cin // G) * K*K, Hout*Wout
        grad_inputs_unfolded = grad_inputs_transposed.reshape(B, Cin*K*K, Hout*Wout) # B, Cin*K*K, Hout*Wout
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=K)
        grad_inputs = fold(grad_inputs_unfolded)

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