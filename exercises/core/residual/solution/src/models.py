# deep learning libraries
import torch

# other libraries
import math
from typing import Any


def forward_prelu(inputs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    This is the forward method of the PReLU.

    Args:
        ctx: context for saving elements for the backward.
        inputs: input tensor. Dimensions: [*].
        a: parameter of PReLU. Dimensions: [0].

    Returns:
        outputs tensor. Dimensions: [*], same as inputs.
    """

    # compute forward
    outputs = inputs.clone()
    outputs[inputs <= 0] = a * outputs[inputs <= 0]

    return outputs


def backward_prelu(
    grad_output: torch.Tensor, inputs: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This method is the backward of the PReLU.

    Args:
        grad_output: outputs gradients. Dimensions: [*].
        inputs: input tensor. Dimensions: [*].
        a: parameter of PReLU. Dimensions: [0].

    Returns:
        inputs gradients. Dimensions: [*], same as the grad_output.
        a gradients. Dimensions: [0], same as the a parameter.
    """

    # compute input gradients
    grad_input: torch.Tensor = torch.ones_like(inputs)
    grad_input[inputs <= 0] = a
    grad_input *= grad_output

    # compute a gradients
    grad_a: torch.Tensor = torch.sum(inputs[inputs <= 0] * grad_output[inputs <= 0])

    return grad_input, grad_a


class PReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    @torch.no_grad()
    def forward(ctx: Any, inputs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the PReLU.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].
            a: parameter of PReLU. Dimensions: [0].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # save tensors for the backward
        ctx.save_for_backward(inputs, a)

        # compute forward
        outputs = forward_prelu(inputs, a)

        return outputs

    @staticmethod
    @torch.no_grad()
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """
        This method is the backward of the PReLU.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
            a gradients. Dimension: [0].
        """

        # load tensors from the forward
        inputs, a = ctx.saved_tensors

        grad_input: torch.Tensor
        grad_a: torch.Tensor
        grad_input, grad_a = backward_prelu(grad_output, inputs, a)

        return grad_input, grad_a


class PReLU(torch.nn.Module):
    """
    This is the class that represents the PReLU Layer.
    """

    def __init__(self, init: float = 0.25) -> None:
        """
        This method is the constructor of the PReLU layer.
        """

        # call super class constructor
        super().__init__()

        self.a: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(init))

        self.fn = PReLUFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs, self.a)


class ResidualFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Residual.
    """

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the residual.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dimension].
            weight: weights tensor.
                Dimensions: [input dimension, input dimension].
            bias: bias tensor. Dimensions: [input dimension].
            a: parameter of PReLU. Dimensions: [0].

        Returns:
            outputs tensor. Dimensions: [batch, input dimension].
        """

        # save elements for backward
        ctx.save_for_backward(inputs, weight, bias, a)

        # compute forward
        outputs = torch.matmul(inputs, weight.T) + bias.unsqueeze(0)
        outputs = forward_prelu(outputs, a)
        outputs = outputs + inputs

        return outputs

    @staticmethod
    @torch.no_grad()
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        """
        This method is the backward of the residual.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients.
                Dimensions: [batch, input dimension].

        Returns:
            inputs gradients. Dimension: [batch, input dimension].
            weights gradients.
                Dimension: [input dimension, input dimension].
            bias gradients. Dimension: [input dimension].
            a gradients. Dimension: [0].
        """

        # load elements from forward
        inputs, weight, bias, a = ctx.saved_tensors

        # compute gradients of prelu
        inputs_prelu = torch.matmul(inputs, weight.T) + bias.unsqueeze(0)
        grad_input_prelu, grad_a = backward_prelu(grad_output, inputs_prelu, a)

        # compute gradients
        grad_input = torch.matmul(grad_input_prelu, weight) + grad_output
        grad_weight = torch.matmul(inputs.T, grad_input_prelu).T
        grad_bias = torch.matmul(
            torch.ones_like(inputs[:, 0]).unsqueeze(0), grad_input_prelu
        ).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_a


class Residual(torch.nn.Module):
    """
    This is the class that represents the Residual Layer.
    """

    def __init__(self, input_dim: int, init: float = 0.25) -> None:
        """
        This method is the constructor of the Residual layer.

        Args:
            input_dim: input dimension.
            a: parameter of prelu.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(input_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(input_dim))
        self.a: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(init))

        # init parameters corectly
        self.reset_parameters()

        self.fn = ResidualFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dimension].

        Returns:
            outputs tensor. Dimensions: [batch, input dimension].
        """

        return self.fn(inputs, self.weight, self.bias, self.a)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None
