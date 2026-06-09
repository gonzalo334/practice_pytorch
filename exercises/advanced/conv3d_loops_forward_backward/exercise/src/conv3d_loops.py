# deep learning libraries
import torch

# other libraries
import math
from typing import Any


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


def _conv_output_size(
    input_size: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
    padding: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> tuple[int, int, int]:
    output_depth = (
        input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    output_height = (
        input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    output_width = (
        input_size[2] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1
    ) // stride[2] + 1
    return output_depth, output_height, output_width


class Conv3dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of Conv3d
    using explicit loops.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        This is the forward method of the Conv3d layer.

        Args:
            inputs: input tensor with shape
                ``[batch, in_channels, depth, height, width]``.
            weight: convolution weights with shape
                ``[out_channels, in_channels, kernel_depth, kernel_height,
                kernel_width]``.
            bias: bias tensor with shape ``[out_channels]``.
            stride: convolution stride for depth, height, and width.
            padding: zero padding for depth, height, and width.
            dilation: kernel dilation for depth, height, and width.

        Returns:
            Output tensor with shape
            ``[batch, out_channels, output_depth, output_height, output_width]``.
        """

        # TODO
        B, Cin, D, H, W = inputs.shape
        Cout, Cin, Kd, Kh, Kw = weight.shape
        Cout = bias.shape
        Sd, Sh, Sw = 1, 1, 1
        Pd, Ph, Pw = 0, 0, 0
        Dd, Dh, Dw = 1, 1, 1

        


    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This method is the backward of the Conv3d layer.
        """

        # TODO


class Conv3d(torch.nn.Module):
    """
    This is the class that represents a Conv3d layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)

        self.weight = torch.nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
                dtype=dtype,
            )
        )
        self.bias = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype))

        self.reset_parameters()
        self.fn = Conv3dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)
