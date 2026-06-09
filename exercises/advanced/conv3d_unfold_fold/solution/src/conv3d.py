# deep learning libraries
import torch
import torch.nn.functional as F

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
    dilation: int,
    padding: int,
    stride: int,
) -> tuple[int, int, int]:
    depth = (
        (input_size[0] + 2 * padding - dilation * (kernel_size[0] - 1) - 1)
        // stride
        + 1
    )
    height = (
        (input_size[1] + 2 * padding - dilation * (kernel_size[1] - 1) - 1)
        // stride
        + 1
    )
    width = (
        (input_size[2] + 2 * padding - dilation * (kernel_size[2] - 1) - 1)
        // stride
        + 1
    )
    return depth, height, width


def unfold3d(
    inputs: torch.Tensor,
    kernel_size: tuple[int, int, int],
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """
    This operation computes the unfold operation for 3d convolutions.
    """

    B, Cin, D, H, W = inputs.shape
    Kd, Kh, Kw = kernel_size

    P = padding
    S = stride
    Dil = dilation

    Dout = (D + 2 * P - Dil * (Kd - 1) - 1) // S + 1
    Hout = (H + 2 * P - Dil * (Kh - 1) - 1) // S + 1
    Wout = (W + 2 * P - Dil * (Kw - 1) - 1) // S + 1

    # [B, Cin, D, H, W] -> [B, D, Cin, H, W] -> [B*D, Cin, H, W]
    inputs_hw = inputs.transpose(1, 2).contiguous().view(B * D, Cin, H, W)

    # Unfold height and width
    # [B*D, Cin*Kh*Kw, Hout*Wout]
    inputs_hw_unfolded = F.unfold(
        inputs_hw,
        kernel_size=(Kh, Kw),
        dilation=Dil,
        padding=P,
        stride=S,
    )

    # [B*D, Cin*Kh*Kw, Hout*Wout]
    # -> [B, D, Cin, Kh, Kw, Hout, Wout]
    inputs_hw_unfolded = inputs_hw_unfolded.view(
        B, D, Cin, Kh, Kw, Hout, Wout
    )

    # Prepare depth as the height dimension of a fake 2D image.
    # We want channels = Cin*Kh*Kw*Hout*Wout, height = D, width = 1
    # [B, D, Cin, Kh, Kw, Hout, Wout]
    # -> [B, Cin, Kh, Kw, Hout, Wout, D]
    # -> [B, Cin*Kh*Kw*Hout*Wout, D, 1]
    inputs_depth = (
        inputs_hw_unfolded.permute(0, 2, 3, 4, 5, 6, 1)
        .contiguous()
        .view(B, Cin * Kh * Kw * Hout * Wout, D, 1)
    )

    # Unfold depth
    # [B, Cin*Kh*Kw*Hout*Wout*Kd, Dout]
    inputs_depth_unfolded = F.unfold(
        inputs_depth,
        kernel_size=(Kd, 1),
        dilation=(Dil, 1),
        padding=(P, 0),
        stride=(S, 1),
    )

    # [B, Cin*Kh*Kw*Hout*Wout*Kd, Dout]
    # -> [B, Cin, Kh, Kw, Hout, Wout, Kd, Dout]
    inputs_depth_unfolded = inputs_depth_unfolded.view(
        B, Cin, Kh, Kw, Hout, Wout, Kd, Dout
    )

    # Reorder to Conv3d-compatible flattening:
    # [B, Cin, Kd, Kh, Kw, Dout, Hout, Wout]
    inputs_depth_unfolded = inputs_depth_unfolded.permute(
        0, 1, 6, 2, 3, 7, 4, 5
    ).contiguous()

    # Final:
    # [B, Cin*Kd*Kh*Kw, Dout*Hout*Wout]
    outputs = inputs_depth_unfolded.view(
        B, Cin * Kd * Kh * Kw, Dout * Hout * Wout
    )

    return outputs



def fold3d(
    inputs: torch.Tensor,
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """
    This operation computes the fold operation for 3d convolutions.
    """

    batch_size, flattened_channels, _ = inputs.shape
    kernel_volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    channels = flattened_channels // kernel_volume
    output_depth, output_height, output_width = output_size
    windows_depth, windows_height, windows_width = _conv_output_size(
        output_size, kernel_size, dilation, padding, stride
    )
    padded_depth = output_depth + 2 * padding

    inputs = inputs.reshape(
        batch_size,
        channels,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        windows_depth,
        windows_height * windows_width,
    )

    height_width_columns = inputs.permute(0, 2, 5, 1, 3, 4, 6).reshape(
        batch_size * kernel_size[0] * windows_depth,
        channels * kernel_size[1] * kernel_size[2],
        windows_height * windows_width,
    )
    height_width_folded = F.fold(
        height_width_columns,
        output_size=(output_height, output_width),
        kernel_size=kernel_size[1:],
        dilation=dilation,
        padding=padding,
        stride=stride,
    ).reshape(
        batch_size,
        kernel_size[0],
        windows_depth,
        channels,
        output_height,
        output_width,
    )
    height_width_folded = height_width_folded.permute(0, 3, 1, 2, 4, 5).reshape(
        batch_size,
        channels,
        kernel_size[0] * windows_depth,
        output_height,
        output_width,
    )

    kernel_depth_idx = torch.arange(kernel_size[0], device=inputs.device).view(-1, 1)
    window_depth_idx = torch.arange(windows_depth, device=inputs.device).view(1, -1)
    depth_idx = kernel_depth_idx * dilation + window_depth_idx * stride
    depth_idx = depth_idx.reshape(1, 1, -1, 1, 1).expand_as(height_width_folded)

    folded = inputs.new_zeros(
        batch_size, channels, padded_depth, output_height, output_width
    )
    folded.scatter_add_(2, depth_idx, height_width_folded)

    return folded[:, :, padding : padding + output_depth]


class Conv3dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv3d
    layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dilation: int,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.
        """

        output_size = _conv_output_size(
            inputs.shape[2:], weight.shape[2:], dilation, padding, stride
        )
        inputs_unfolded = unfold3d(
            inputs,
            weight.shape[2:],
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        outputs = torch.matmul(
            weight.view(weight.shape[0], -1).unsqueeze(0),
            inputs_unfolded,
        )
        outputs = outputs.view(inputs.shape[0], weight.shape[0], *output_size)
        outputs = outputs + bias.view(1, -1, 1, 1, 1)

        ctx.save_for_backward(inputs, weight)
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This is the backward of the layer.
        """

        inputs, weight = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(
            grad_output.shape[0], grad_output.shape[1], -1
        )
        inputs_unfolded = unfold3d(
            inputs,
            weight.shape[2:],
            dilation=ctx.dilation,
            padding=ctx.padding,
            stride=ctx.stride,
        )

        grad_inputs_unfolded = torch.matmul(
            weight.view(weight.shape[0], -1).t().unsqueeze(0),
            grad_output_flat,
        )
        grad_inputs = fold3d(
            grad_inputs_unfolded,
            inputs.shape[2:],
            weight.shape[2:],
            dilation=ctx.dilation,
            padding=ctx.padding,
            stride=ctx.stride,
        )
        grad_weight = torch.matmul(
            grad_output_flat,
            inputs_unfolded.transpose(1, 2),
        ).sum(dim=0)
        grad_weight = grad_weight.view_as(weight)
        grad_bias = grad_output.sum(dim=(0, 2, 3, 4))

        return grad_inputs, grad_weight, grad_bias, None, None, None


class Conv3d(torch.nn.Module):
    """
    This is the class that represents the Conv3d layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int, int],
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Conv3d layer.
        """

        super().__init__()
        kernel_size = _triple(kernel_size)
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.weight = torch.nn.Parameter(
            torch.empty(
                output_channels,
                input_channels,
                kernel_size[0],
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.bias = torch.nn.Parameter(torch.empty(output_channels))

        self.reset_parameters()
        self.fn = Conv3dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.
        """

        return self.fn(
            inputs, self.weight, self.bias, self.dilation, self.padding, self.stride
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)
