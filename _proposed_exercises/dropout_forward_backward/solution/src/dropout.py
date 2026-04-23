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

        if not training or p == 0:
            mask = torch.ones_like(inputs)
            ctx.save_for_backward(mask)
            return inputs.clone()

        mask = (torch.rand_like(inputs) > p).to(inputs.dtype) / (1.0 - p)
        outputs = inputs * mask
        ctx.save_for_backward(mask)

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        This method is the backward of Dropout.
        """

        (mask,) = ctx.saved_tensors
        grad_inputs = grad_outputs * mask

        return grad_inputs, None, None


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

