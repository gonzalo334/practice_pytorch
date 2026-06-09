# deep learning libraries
import torch

# other libraries
from typing import Any


class MaxoutFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of Maxout.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the Maxout layer.

        Args:
            ctx: context used to save tensors for the backward pass.
            inputs: input tensor with shape ``[batch, input_dim]``.
            weights: affine weights with shape
                ``[num_units, output_dim, input_dim]``.
            bias: affine bias with shape ``[num_units, output_dim]``.

        Returns:
            Output tensor with shape ``[batch, output_dim]`` containing the
            maximum affine response over the ``num_units`` axis.
        """

        # TODO
        
    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward of the Maxout layer.

        Args:
            ctx: context containing tensors saved during ``forward``.
            grad_outputs: upstream gradients with shape
                ``[batch, output_dim]``.

        Returns:
            Gradients for ``inputs``, ``weights``, and ``bias`` with shapes
            ``[batch, input_dim]``, ``[num_units, output_dim, input_dim]``,
            and ``[num_units, output_dim]``, respectively.
        """

        # TODO

class Maxout(torch.nn.Module):
    """
    This is the class that represents the Maxout layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_units: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = num_units
        self.weights = torch.nn.Parameter(
            torch.empty(num_units, output_dim, input_dim, dtype=dtype)
        )
        self.bias = torch.nn.Parameter(torch.empty(num_units, output_dim, dtype=dtype))
        self.reset_parameters()
        self.fn = MaxoutFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Maxout layer output.

        Args:
            inputs: input tensor with shape ``[batch, input_dim]``.

        Returns:
            Output tensor with shape ``[batch, output_dim]``.
        """

        return self.fn(inputs, self.weights, self.bias)

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weights, a=5**0.5)

        fan_in = self.input_dim
        bound = 1 / fan_in**0.5
        torch.nn.init.uniform_(self.bias, -bound, bound)
