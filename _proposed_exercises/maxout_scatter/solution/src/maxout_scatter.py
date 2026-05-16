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
        B, Din = inputs.shape
        K, Dout, Din = weights.shape
        # K, Dout = bias.shape
        inputs_dim = inputs.unsqueeze(1).unsqueeze(1) # B, 1, 1, Din
        z_units = inputs_dim @ weights.transpose(1,2) + bias.unsqueeze(1) # B, K, 1, Dout
        z_units = z_units.view(B, K, Dout)
        indexes = torch.argmax(z_units, 1, keepdim=True)
        outputs = torch.gather(z_units, 1, indexes)
        ctx.save_for_backward(inputs, weights, bias, indexes, z_units)
        return outputs.squeeze(1)
        
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
        inputs, weights, bias, indexes, z_units = ctx.saved_tensors
        B, Din = inputs.shape
        K, Dout, Din = weights.shape
        # K, Dout = bias.shape
        # B, 1, Dout = indexes.shape
        grad_inputs = torch.zeros_like(inputs)
        grad_weights = torch.zeros_like(weights)
        grad_bias = torch.zeros_like(bias)
        grad_z = torch.zeros_like(z_units) # B, K, Dout

        grad_z = torch.scatter(grad_z, 1, indexes, grad_outputs.unsqueeze(1)).view(B, K*Dout)

        grad_inputs = grad_z @ weights.view(K*Dout, Din)
        grad_weights = (inputs.T.unsqueeze(0) @ grad_z).view(Din, K, Dout).permute(1, 2, 0)
        grad_bias = grad_z.sum(dim=0).view(K, Dout)

        return grad_inputs, grad_weights, grad_bias


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
