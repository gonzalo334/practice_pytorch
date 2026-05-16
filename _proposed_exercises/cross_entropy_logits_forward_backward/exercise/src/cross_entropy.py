# deep learning libraries
import torch

# other libraries
from typing import Any


class CrossEntropyLossFunction(torch.autograd.Function):
    """
    Class for cross entropy loss from logits.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        """
        Forward pass of cross entropy loss.
        """

        # TODO
        raise NotImplementedError

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        Backward pass of cross entropy loss.
        """

        # TODO
        raise NotImplementedError


class CrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy loss computed from logits.
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.fn = CrossEntropyLossFunction.apply

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fn(logits, targets, self.ignore_index, self.reduction)
