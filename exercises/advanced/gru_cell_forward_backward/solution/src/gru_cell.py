# deep learning libraries
import torch

# other libraries
import math
from typing import Any


def _gru_cell_expression(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> torch.Tensor:
    gi = inputs @ weight_ih.T + bias_ih
    gh = hidden @ weight_hh.T + bias_hh

    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)

    reset_gate = torch.sigmoid(i_r + h_r)
    update_gate = torch.sigmoid(i_z + h_z)
    new_gate = torch.tanh(i_n + reset_gate * h_n)

    return new_gate + update_gate * (hidden - new_gate)


class GRUCellFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of GRUCell.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of a GRU cell.
        """

        outputs = _gru_cell_expression(
            inputs, hidden, weight_ih, weight_hh, bias_ih, bias_hh
        )
        ctx.save_for_backward(inputs, hidden, weight_ih, weight_hh, bias_ih, bias_hh)
        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass of a GRU cell.
        """

        inputs, hidden, weight_ih, weight_hh, bias_ih, bias_hh = ctx.saved_tensors
        with torch.enable_grad():
            inputs_grad = inputs.detach().requires_grad_(True)
            hidden_grad = hidden.detach().requires_grad_(True)
            weight_ih_grad = weight_ih.detach().requires_grad_(True)
            weight_hh_grad = weight_hh.detach().requires_grad_(True)
            bias_ih_grad = bias_ih.detach().requires_grad_(True)
            bias_hh_grad = bias_hh.detach().requires_grad_(True)

            outputs = _gru_cell_expression(
                inputs_grad,
                hidden_grad,
                weight_ih_grad,
                weight_hh_grad,
                bias_ih_grad,
                bias_hh_grad,
            )
            grads = torch.autograd.grad(
                outputs,
                (
                    inputs_grad,
                    hidden_grad,
                    weight_ih_grad,
                    weight_hh_grad,
                    bias_ih_grad,
                    bias_hh_grad,
                ),
                grad_outputs,
            )

        return grads


class GRUCell(torch.nn.Module):
    """
    This is the class that represents a single GRU cell.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = torch.nn.Parameter(
            torch.empty(3 * hidden_dim, input_dim, dtype=dtype)
        )
        self.weight_hh = torch.nn.Parameter(
            torch.empty(3 * hidden_dim, hidden_dim, dtype=dtype)
        )
        self.bias_ih = torch.nn.Parameter(torch.empty(3 * hidden_dim, dtype=dtype))
        self.bias_hh = torch.nn.Parameter(torch.empty(3 * hidden_dim, dtype=dtype))

        self.reset_parameters()
        self.fn = GRUCellFunction.apply

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.fn(
            inputs,
            hidden,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.hidden_dim)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -bound, bound)
