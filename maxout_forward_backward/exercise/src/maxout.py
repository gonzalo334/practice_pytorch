# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class MaxoutFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Maxout.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # TODO
        B, Din = inputs.shape
        Dout, Din = weights_first.shape
        # Dout = bias_first.shape
        z1 = inputs @ weights_first.T + bias_first.unsqueeze(0) 
        z2 = inputs @ weights_second.T + bias_second.unsqueeze(0) 
        z = torch.concat((z1.unsqueeze(1), z2.unsqueeze(1)), dim=1)
        outputs, indexes = torch.max(z, dim=1)
        mask = z1 > z2
        ctx.save_for_backward(inputs, weights_first, weights_second, bias_first, bias_second, mask)
        return outputs




    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward of the Maxout.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, output dim].

        Returns:
            inputs gradients. Dimensions: [batch, input dim].
            gradients for the first weights. Dimensions:
                [output dim, input dim].
            gradients for the first bias. Dimensions: [output dim].
            gradient for the second weights. Dimensions: [output dim,
                input dim].
            gradient for the second bias. Dimensions: [output dim].
        """

        # TODO
        inputs, weights_first, weights_second, bias_first, bias_second, mask = ctx.saved_tensors
        B, Din = inputs.shape
        Dout, Din = weights_first.shape
        # B, Dout = grad_output.shape

        grad_inputs = (grad_output*mask) @ weights_first + (grad_output * (~mask)) @ weights_second

        grad_weight_first = (grad_output*mask).T @ inputs
        grad_weight_second = (grad_output*(~mask)).T @ inputs

        grad_bias_first = (grad_output*mask).sum(dim=0)
        grad_bias_second = (grad_output*(~mask)).sum(dim=0)

        return grad_inputs, grad_weight_first, grad_bias_first, grad_weight_second, grad_bias_second





class Maxout(torch.nn.Module):
    """
    This is the class that represents the Maxout Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Maxout layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weights_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )
        self.weights_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        self.fn = MaxoutFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(
            inputs,
            self.weights_first,
            self.bias_first,
            self.weights_second,
            self.bias_second,
        )

    @torch.no_grad()
    def set_parameters(
        self,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> None:
        """
        This function is to set the parameters of the model.

        Args:
            weights_first: weights for the first branch.
            bias_first: bias for the first branch.
            weights_second: weights for the second branch.
            bias_second: bias for the second branch.
        """

        # set attributes
        self.weights_first = torch.nn.Parameter(weights_first)
        self.bias_first = torch.nn.Parameter(bias_first)
        self.weights_second = torch.nn.Parameter(weights_second)
        self.bias_second = torch.nn.Parameter(bias_second)

        return None

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weights_first, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights_second, a=math.sqrt(5))
        if self.bias_first is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_first)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_first, -bound, bound)
            torch.nn.init.uniform_(self.bias_second, -bound, bound)

        return None
