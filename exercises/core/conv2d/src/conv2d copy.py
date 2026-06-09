# Standard libraries
import math
from typing import Any

# 3pps
import torch
import torch.nn.functional as F

class Conv2dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: Weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: Bias of the layer. Dimensions: [output channels].
            padding: padding parameter.
            stride: stride parameter.

        Returns:
            Output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """
        B, Cin, H, W = inputs.shape
        Cout, Cin, K, K = weight.shape
        Hout = (H + 2*padding - K) // stride + 1
        Wout = (W + 2*padding - K) // stride + 1

        inputs_unfolded = torch.nn.functional.unfold(inputs, K, padding=padding, stride=stride) # B, Cin*K*K, Hout*Wout
        weight_windows = weight.view(Cout, Cin*K*K).unsqueeze(0) # 1, Cout, Cin*K*K
        bias = bias.unsqueeze(0).unsqueeze(-1) # 1, Cout, 1
        outputs = weight_windows @ inputs_unfolded + bias # B, Cout, Hout*Wout
        outputs = outputs.view(B, Cout, Hout, Wout)

        ctx.save_for_backward(inputs, weight)   # bias no se necesita ya que su derivada es 1
        ctx.padding = padding
        ctx.stride = stride
        
        return outputs
    
    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: Context for loading elements needed in the backward.
            grad_output: Outputs gradients. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            Inputs gradients. Dimensions: [batch, input channels,
                height, width].
            Weight gradients. Dimensions: [output channels,
                input channels, kernel size, kernel size].
            Bias gradients. Dimensions: [output channels].
            None.
            None.
        """
        inputs, weight = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        B, Cin, H, W = inputs.shape
        Cout, Cin, K, K = weight.shape
        B, Cout, Hout, Wout = grad_output.shape
        
        inputs_unfolded = torch.nn.functional.unfold(inputs, K, padding=padding, stride=stride) # B, Cin*K*K, Hout*Wout
        weight_unfolded = weight.view(Cout, Cin*K*K)    # Cout, Cin*K*K
        grad_output_unfolded = grad_output.view(B, Cout, Hout*Wout) # B, Cout, Hout*Wout

        grad_inputs_unfolded = weight_unfolded.T @ grad_output_unfolded  # B, Cin*K*K, Hout*Wout
        grad_inputs = torch.nn.functional.fold(grad_inputs_unfolded, output_size=(H,W), kernel_size=K, padding=padding, stride=stride) # B, Cin*K*K, Hout*Wout

        grad_weight = grad_output_unfolded @ inputs_unfolded.transpose(1,2) # B, Cout, Cin*K*K 
        grad_weight = grad_weight.sum(dim=0).view(Cout, Cin, K, K)  # Cin, Cout, K, K
        
        grad_bias = grad_output.sum(dim=(0,2,3)) # Cout 

        return grad_inputs, grad_weight, grad_bias, None, None

