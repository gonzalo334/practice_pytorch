# deep learning libraries
import torch

# other libraries
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
