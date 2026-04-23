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

        # compute forward
        outputs_first: torch.Tensor = torch.matmul(
            inputs, weights_first.T
        ) + bias_first.unsqueeze(0)
        outputs_second: torch.Tensor = torch.matmul(
            inputs, weights_second.T
        ) + bias_second.unsqueeze(0)

        # compute final outputs
        outputs: torch.Tensor = outputs_second.clone()
        indexes: torch.Tensor = outputs_second <= outputs_first
        outputs[indexes] = outputs_first[indexes]

        # save tensors for the backward
        ctx.save_for_backward(
            inputs, weights_first, bias_first, weights_second, bias_second, indexes
        )

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

        # load tensors from the forward
        (
            inputs,
            weights_first,
            bias_first,
            weights_second,
            bias_second,
            indexes,
        ) = ctx.saved_tensors

        # compute grad outputs for each branch
        grad_outputs_first: torch.Tensor = grad_output.clone()
        grad_outputs_second: torch.Tensor = grad_output.clone()
        grad_outputs_first[~indexes] = 0
        grad_outputs_second[indexes] = 0

        # compuye grad inputs
        grad_inputs_first = torch.matmul(grad_outputs_first, weights_first)
        grad_inputs_second = torch.matmul(grad_outputs_second, weights_second)
        grad_inputs = grad_inputs_first + grad_inputs_second

        # compute weights and bias gradients
        grad_weight_first = torch.matmul(inputs.T, grad_outputs_first).T
        grad_weight_second = torch.matmul(inputs.T, grad_outputs_second).T
        grad_bias_first = torch.matmul(
            torch.ones_like(inputs[:, 0]).unsqueeze(0), grad_outputs_first
        ).squeeze(0)
        grad_bias_second = torch.matmul(
            torch.ones_like(inputs[:, 0]).unsqueeze(0), grad_outputs_second
        ).squeeze(0)

        return (
            grad_inputs,
            grad_weight_first,
            grad_bias_first,
            grad_weight_second,
            grad_bias_second,
        )


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
