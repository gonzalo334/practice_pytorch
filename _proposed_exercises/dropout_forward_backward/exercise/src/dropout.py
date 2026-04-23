# deep learning libraries
import torch

# other libraries
from typing import Any


class DropoutFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of Dropout.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        p: float,
        training: bool,
    ) -> torch.Tensor:
        """
        This is the forward method of Dropout.
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        This method is the backward of Dropout.
        """

        # TODO


class Dropout(torch.nn.Module):
    """
    This is the class that represents the Dropout layer.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("p must satisfy 0 <= p < 1")
        self.p = p
        self.fn = DropoutFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.p, self.training)

