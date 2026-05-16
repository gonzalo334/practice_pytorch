# deep learning libraries
import torch

# other libraries
from typing import Any


class InstanceNorm2dFunction(torch.autograd.Function):
    """
    Class for the implementation of InstanceNorm2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Forward pass of InstanceNorm2d.
        """

        mean = inputs.mean(dim=(2, 3), keepdim=True)
        var = inputs.var(dim=(2, 3), unbiased=False, keepdim=True)
        inv_std = torch.rsqrt(var + eps)
        normalized = (inputs - mean) * inv_std
        outputs = normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

        ctx.save_for_backward(normalized, inv_std, weight)

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of InstanceNorm2d.
        """

        normalized, inv_std, weight = ctx.saved_tensors
        elements = normalized.shape[2] * normalized.shape[3]

        grad_normalized = grad_outputs * weight.view(1, -1, 1, 1)
        grad_sum = grad_normalized.sum(dim=(2, 3), keepdim=True)
        grad_normalized_sum = (grad_normalized * normalized).sum(
            dim=(2, 3), keepdim=True
        )
        grad_inputs = (
            grad_normalized
            - grad_sum / elements
            - normalized * grad_normalized_sum / elements
        ) * inv_std

        grad_weight = (grad_outputs * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_outputs.sum(dim=(0, 2, 3))

        return grad_inputs, grad_weight, grad_bias, None


class InstanceNorm2d(torch.nn.Module):
    """
    This is the class that represents InstanceNorm2d.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(num_features, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.empty(num_features, dtype=dtype))
        self.reset_parameters()
        self.fn = InstanceNorm2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.weight, self.bias, self.eps)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
