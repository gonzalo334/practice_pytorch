"""
This module contains the code for the custom BatchNorm2D.
"""

# Standard libraries
from typing import Any

# 3pps
import torch


class BatchNorm2dFunction(torch.autograd.Function):
    """
    This class implements the forward and backward passes of the 
    BatchNorm2d.

    Attributes:
        ctx: Context for saving elements for the backward.
    """
    
    # Define attributes
    ctx: Any
    
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        eps: float,
        negative_slope: float,
    ) -> torch.Tensor:
        """
        This is the forward method of the BatchNorm2d.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].
            running_mean: Tensor for the running mean. Dimensions:
                [channels].
            running_var: Tensor for the running std. Dimensions:
                [channels].
            eps: Epsilon for the denominator.
            negative_slope: Negative slope for the LeakyReLU.

        Returns:
            Outputs tensor. Dimensions: [batch, channels, height,
                width], same as inputs.
        """

        # Compute numerator and denominator
        numerator = inputs - running_mean.view(1, -1, 1, 1)
        outputs = numerator.clone()
        denominator = torch.sqrt(running_var.view(1, -1, 1, 1) + eps)

        # Apply leaky relu
        outputs[numerator < 0] = outputs[numerator < 0] * negative_slope

        # Compute outputs
        outputs = outputs / denominator

        # Save for backward
        ctx.save_for_backward(
            inputs, numerator, denominator, torch.tensor(negative_slope)
        )

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """
        This method is the backward pass of the layer.

        Args:
            grad_outputs: Outputs gradients. Dimensions: 
                [batch size, number of channels, height, width].

        Returns:
            Inputs gradients. Dimensions: [batch size, 
                number of channels, height, width].
            None.
            None.
            None.
            None.
        """
        
        # Load tensors from the forward
        inputs, numerator, denominator, negative_slope = ctx.saved_tensors

        # Compute grad inputs
        grad_inputs: torch.Tensor = grad_outputs / denominator

        # Apply leaky relu
        grad_inputs[numerator < 0] *= negative_slope

        return grad_inputs, None, None, None, None


class BatchNorm2d(torch.nn.Module):
    """
    This class implements the BatchNorm2d.

    Attributes:
        num_features: Number of channels.
        running_mean: Tensor with the mean to apply. Dimensions: 
            [number of channels].
        running_var: Tensor with the var to apply. Dimensions: 
            [number of channels].
        negative_slope: Negative slope of the LeakyReLU.
    """
    
    # Define attributes
    num_features: int
    running_mean: torch.Tensor
    running_var: torch.Tensor
    eps: float
    negative_slope: float
    
    def __init__(self, num_features: int, negative_slope: float = 0.01) -> None:
        """
        This method is the constructor of the class.

        Args:
            num_features: Number of channels.
            negative_slope: Negative slope of the LeakyReLU. 
                Defaults to 0.01.

        Returns:
            None.
        """
        
        # Call super class constructor
        super().__init__()

        # Set attributes
        self.num_features = num_features
        self.running_mean = torch.zeros(num_features, dtype=torch.double)
        self.running_var = torch.zeros(num_features, dtype=torch.double)
        self.eps = 1e-5
        self.negative_slope = negative_slope

        # Set function
        self.fn = BatchNorm2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, 
                number of channels, height, width].

        Returns:
            Outputs tensor. Dimensions: [batch size, 
                number of channels, height, width].
        """
        
        return self.fn(
            inputs,
            self.running_mean,
            self.running_var,
            self.eps,
            self.negative_slope,
        )
