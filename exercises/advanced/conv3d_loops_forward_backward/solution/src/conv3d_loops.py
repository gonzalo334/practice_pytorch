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
        """

        batch_size, in_channels, _, _, _ = inputs.shape
        out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
        output_depth, output_height, output_width = _conv_output_size(
            inputs.shape[2:], weight.shape[2:], dilation, padding, stride
        )

        outputs = inputs.new_empty(
            batch_size, out_channels, output_depth, output_height, output_width
        )

        for batch in range(batch_size):
            for out_channel in range(out_channels):
                for out_depth in range(output_depth):
                    for out_height in range(output_height):
                        for out_width in range(output_width):
                            value = bias[out_channel].clone()
                            for in_channel in range(in_channels):
                                for k_depth in range(kernel_depth):
                                    input_depth = (
                                        out_depth * stride[0]
                                        + k_depth * dilation[0]
                                        - padding[0]
                                    )
                                    if input_depth < 0 or input_depth >= inputs.shape[2]:
                                        continue
                                    for k_height in range(kernel_height):
                                        input_height = (
                                            out_height * stride[1]
                                            + k_height * dilation[1]
                                            - padding[1]
                                        )
                                        if (
                                            input_height < 0
                                            or input_height >= inputs.shape[3]
                                        ):
                                            continue
                                        for k_width in range(kernel_width):
                                            input_width = (
                                                out_width * stride[2]
                                                + k_width * dilation[2]
                                                - padding[2]
                                            )
                                            if (
                                                input_width < 0
                                                or input_width >= inputs.shape[4]
                                            ):
                                                continue
                                            value = value + (
                                                inputs[
                                                    batch,
                                                    in_channel,
                                                    input_depth,
                                                    input_height,
                                                    input_width,
                                                ]
                                                * weight[
                                                    out_channel,
                                                    in_channel,
                                                    k_depth,
                                                    k_height,
                                                    k_width,
                                                ]
                                            )
                            outputs[
                                batch, out_channel, out_depth, out_height, out_width
                            ] = value

        ctx.save_for_backward(inputs, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This method is the backward of the Conv3d layer.
        """

        inputs, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation

        batch_size, in_channels, _, _, _ = inputs.shape
        out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
        output_depth, output_height, output_width = grad_outputs.shape[2:]

        grad_inputs = torch.zeros_like(inputs)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros(out_channels, dtype=inputs.dtype, device=inputs.device)

        for batch in range(batch_size):
            for out_channel in range(out_channels):
                for out_depth in range(output_depth):
                    for out_height in range(output_height):
                        for out_width in range(output_width):
                            grad_value = grad_outputs[
                                batch, out_channel, out_depth, out_height, out_width
                            ]
                            grad_bias[out_channel] = grad_bias[out_channel] + grad_value
                            for in_channel in range(in_channels):
                                for k_depth in range(kernel_depth):
                                    input_depth = (
                                        out_depth * stride[0]
                                        + k_depth * dilation[0]
                                        - padding[0]
                                    )
                                    if input_depth < 0 or input_depth >= inputs.shape[2]:
                                        continue
                                    for k_height in range(kernel_height):
                                        input_height = (
                                            out_height * stride[1]
                                            + k_height * dilation[1]
                                            - padding[1]
                                        )
                                        if (
                                            input_height < 0
                                            or input_height >= inputs.shape[3]
                                        ):
                                            continue
                                        for k_width in range(kernel_width):
                                            input_width = (
                                                out_width * stride[2]
                                                + k_width * dilation[2]
                                                - padding[2]
                                            )
                                            if (
                                                input_width < 0
                                                or input_width >= inputs.shape[4]
                                            ):
                                                continue
                                            grad_inputs[
                                                batch,
                                                in_channel,
                                                input_depth,
                                                input_height,
                                                input_width,
                                            ] = (
                                                grad_inputs[
                                                    batch,
                                                    in_channel,
                                                    input_depth,
                                                    input_height,
                                                    input_width,
                                                ]
                                                + grad_value
                                                * weight[
                                                    out_channel,
                                                    in_channel,
                                                    k_depth,
                                                    k_height,
                                                    k_width,
                                                ]
                                            )
                                            grad_weight[
                                                out_channel,
                                                in_channel,
                                                k_depth,
                                                k_height,
                                                k_width,
                                            ] = (
                                                grad_weight[
                                                    out_channel,
                                                    in_channel,
                                                    k_depth,
                                                    k_height,
                                                    k_width,
                                                ]
                                                + grad_value
                                                * inputs[
                                                    batch,
                                                    in_channel,
                                                    input_depth,
                                                    input_height,
                                                    input_width,
                                                ]
                                            )

        return grad_inputs, grad_weight, grad_bias, None, None, None


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

