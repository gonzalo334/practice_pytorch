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

        shifted = logits - logits.max(dim=1, keepdim=True).values
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
        valid = targets != ignore_index

        losses = torch.zeros(
            targets.shape[0], dtype=logits.dtype, device=logits.device
        )
        losses[valid] = -log_probs[valid, targets[valid]]

        valid_count = valid.sum().clamp_min(1)
        ctx.save_for_backward(torch.exp(log_probs), targets, valid)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.valid_count = valid_count

        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.sum() / valid_count
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        Backward pass of cross entropy loss.
        """

        probs, targets, valid = ctx.saved_tensors
        grad_logits = probs.clone()
        grad_logits[valid, targets[valid]] -= 1
        grad_logits[~valid] = 0

        if ctx.reduction == "mean":
            grad_logits = grad_logits / ctx.valid_count

        if ctx.reduction == "none":
            grad_logits = grad_logits * grad_outputs.view(-1, 1)
        else:
            grad_logits = grad_logits * grad_outputs

        return grad_logits, None, None, None


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
