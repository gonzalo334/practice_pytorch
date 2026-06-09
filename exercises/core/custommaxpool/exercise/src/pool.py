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
        K, G = kernel_size, num_groups
        Hout, Wout = H - K + 1, W - K + 1
        inputs_unfolded = F.unfold(inputs, (K, K)).unsqueeze(1) # B, 1, Cin*K*K, Hout*Wout
        inputs_windows = inputs_unfolded.view(B, G, (Cin//G) * K*K, Hout, Wout)
        outputs, indexes = torch.max(inputs_windows, dim=2) 
        ctx.save_for_backward(inputs, indexes)
        ctx.K = K
        ctx.G = G
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
        inputs, indexes = ctx.saved_tensors
        G = ctx.G
        K = ctx.K
        B, Cin, H, W = inputs.shape
        Hout, Wout = H - K + 1, W - K + 1
        # B, G, Hout, Wout = grad_outputs.shape
        inputs_unfolded = F.unfold(inputs, (K, K)).unsqueeze(1) # B, 1, Cin*K*K, Hout*Wout
        inputs_windows = inputs_unfolded.view(B, G, (Cin//G) * K*K, Hout, Wout)
        indexes_one_hot = F.one_hot(indexes, num_classes=(Cin//G)*K*K).to(inputs.dtype) # B, G, Hout, Wout, (Cin//G)*K*K
        grad_windows = torch.zeros_like(inputs_windows, dtype=inputs.dtype)
        grad_inputs_unfolded = torch.scatter(grad_windows, 2, indexes.unsqueeze(2), grad_outputs.unsqueeze(2)) # B, G, Hout, Wout, (Cin//G)*K*K
        # grad_inputs_unfolded = indexes_one_hot * grad_outputs.unsqueeze(-1) # B, G, Hout, Wout, (Cin//G)*K*K
        # grad_inputs_unfolded = torch.permute(grad_inputs_unfolded, (0, 1, 4, 2, 3)).view(B*G, (Cin//G)*K*K, Hout*Wout)
        grad_inputs_unfolded = grad_inputs_unfolded.view(B*G, (Cin//G)*K*K, Hout*Wout)
        grad_inputs = F.fold(grad_inputs_unfolded, (H, W), (K, K)).view(B, Cin, H, W)
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
