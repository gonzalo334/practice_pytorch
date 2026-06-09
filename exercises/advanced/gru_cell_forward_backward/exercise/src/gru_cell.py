# deep learning libraries
import torch

# other libraries
import math
from typing import Any


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

        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass of a GRU cell.
        """

        # TODO
        raise NotImplementedError


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
