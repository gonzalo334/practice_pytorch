# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math
from typing import Any


class GroupedConv2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    grouped Conv2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the grouped Conv2d layer.
        """

        outputs = F.conv2d(
            inputs,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        ctx.save_for_backward(inputs, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        """
        This method is the backward of the grouped Conv2d layer.
        """

        inputs, weight, bias = ctx.saved_tensors
        with torch.enable_grad():
            inputs_grad = inputs.detach().requires_grad_(True)
            weight_grad = weight.detach().requires_grad_(True)
            bias_grad = bias.detach().requires_grad_(True)
            outputs = F.conv2d(
                inputs_grad,
                weight_grad,
                bias_grad,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups,
            )
            grad_inputs, grad_weight, grad_bias = torch.autograd.grad(
                outputs,
                (inputs_grad, weight_grad, bias_grad),
                grad_outputs,
            )

        return grad_inputs, grad_weight, grad_bias, None, None, None, None


class GroupedConv2d(torch.nn.Module):
    """
    This is the class that represents a grouped Conv2d layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        self.groups = groups

        self.weight = torch.nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                dtype=dtype,
            )
        )
        self.bias = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype))

        self.reset_parameters()
        self.fn = GroupedConv2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, tuple):
            return value
        return (value, value)

