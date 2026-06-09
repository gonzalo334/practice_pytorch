# deep learning libraries
import torch

# other libraries
from typing import Any


class EmbeddingBagFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of EmbeddingBag.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        offsets: torch.Tensor,
        weight: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """
        Forward pass of EmbeddingBag.
        """

        num_bags = offsets.numel()
        outputs = torch.zeros(
            num_bags, weight.shape[1], dtype=weight.dtype, device=weight.device
        )
        bag_sizes = torch.zeros(num_bags, dtype=weight.dtype, device=weight.device)

        for bag in range(num_bags):
            start = int(offsets[bag].item())
            end = int(offsets[bag + 1].item()) if bag + 1 < num_bags else inputs.numel()
            bag_inputs = inputs[start:end]
            if bag_inputs.numel() > 0:
                outputs[bag] = weight[bag_inputs].sum(dim=0)
                bag_sizes[bag] = bag_inputs.numel()

        if mode == "mean":
            outputs = outputs / bag_sizes.clamp_min(1).unsqueeze(1)
        elif mode != "sum":
            raise ValueError("mode must be 'sum' or 'mean'")

        ctx.save_for_backward(inputs, offsets, weight, bag_sizes)
        ctx.mode = mode

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, None, torch.Tensor, None]:
        """
        Backward pass of EmbeddingBag.
        """

        inputs, offsets, weight, bag_sizes = ctx.saved_tensors
        grad_weight = torch.zeros_like(weight)

        for bag in range(offsets.numel()):
            start = int(offsets[bag].item())
            end = int(offsets[bag + 1].item()) if bag + 1 < offsets.numel() else inputs.numel()
            if end == start:
                continue
            grad_bag = grad_outputs[bag]
            if ctx.mode == "mean":
                grad_bag = grad_bag / bag_sizes[bag]
            grad_weight.index_add_(0, inputs[start:end], grad_bag.expand(end - start, -1))

        return None, None, grad_weight, None


class EmbeddingBag(torch.nn.Module):
    """
    This is the class that represents the EmbeddingBag layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "mean",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        )
        self.mode = mode
        self.reset_parameters()
        self.fn = EmbeddingBagFunction.apply

    def forward(self, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, offsets, self.weight, self.mode)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
