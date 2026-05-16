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


def unfold3d(
    inputs: torch.Tensor,
    kernel_size: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    stride: int | tuple[int, int, int] = 1,
) -> torch.Tensor:
    """
    This operation computes the unfold operation for 3d convolutions.

    Args:
        inputs: input tensor. Dimensions: [batch, input channels,
            input depth, input height, input width].
        kernel_size: kernel size of the unfold operation.
        dilation: dilation of the unfold operation. Defaults to 1.
        padding: padding of the unfold operation. Defaults to 0.
        stride: stride of the unfold operation. Defaults to 1.

    Returns:
        outputs tensor. Dimensions: [batch,
            input channels * kernel depth * kernel height * kernel width,
            number of windows].
    """

    # TODO
    B, Cin, D, H, W = inputs.shape
    Kd, Kh, Kw = kernel_size
    Hout = (H + 2*padding - dilation * (Kh - 1) - 1) // stride + 1
    Wout = (W + 2*padding - dilation * (Kw - 1) - 1) // stride + 1
    Dout = (D + 2*padding - dilation * (Kd - 1) - 1) // stride + 1
    inputs_hw = inputs.transpose(1,2).contiguous().view(B*D, Cin, H, W) # B*D, Cin, H, W
    inputs_hw_unfolded = F.unfold(inputs_hw, (Kh, Kw), dilation, padding, stride) # B*D, Cin*Kh*Kw, Hout*Wout
    inputs_hw_unfolded = inputs_hw_unfolded.view(B, D, Cin, Kh, Kw, Hout, Wout).permute(0,2,3,4,5,6,1).view(B, Cin*Kh*Kw*Hout*Wout, D, 1) # B, Cin*Kh*Kw*Hout*Wout, D, 1
    inputs_hwd_unfolded = F.unfold(inputs_hw_unfolded, (Kd, 1), (dilation, 1), (padding, 0), (stride, 1)) # B, Cin*Kh*Kw*Hout*Wout*Kd, Dout, 1
    inputs_hwd_unfolded = inputs_hwd_unfolded.view(B, Cin, Kh, Kw, Hout, Wout, Kd, Dout).permute(0,1,6,2,3,7,4,5).contiguous() # B, Cin, Kd, Kh, Kw, Dout, Hout, Wout
    inputs_unfolded = inputs_hwd_unfolded.view(B, Cin*Kd*Kh*Kw, Dout*Hout*Wout) # B, Cin*Kh*Kw*Kd, Hout*Wout*Dout
    return inputs_unfolded

    

def fold3d(
    inputs: torch.Tensor,
    output_size: tuple[int, int, int],
    kernel_size: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    stride: int | tuple[int, int, int] = 1,
) -> torch.Tensor:
    """
    This operation computes the fold operation for 3d convolutions.

    Args:
        inputs: input tensor. Dimensions: [batch,
            channels * kernel depth * kernel height * kernel width,
            number of windows].
        output_size: output volume size as (depth, height, width).
        kernel_size: kernel size to use in the fold operation.
        dilation: dilation to use in the fold operation.
        padding: padding to use in the fold operation.
        stride: stride to use in the fold operation.

    Returns:
        output tensor. Dimensions: [batch, channels,
            output depth, output height, output width].
    """

    # TODO
    # Esta función está mal planteada ya que no entiendo bien por qué pero hay que hacer primero el fold D y luego el fold H, W
    # B, C_Kd_Kh_Kw, Hout_Wout_Dout = inputs.shape
    # Kd, Kh, Kw = kernel_size
    # D, H, W = output_size
    # Hout = (H + 2*padding - dilation * (Kh - 1) - 1) // stride + 1
    # Wout = (W + 2*padding - dilation * (Kw - 1) - 1) // stride + 1
    # Dout = (D + 2*padding - dilation * (Kd - 1) - 1) // stride + 1
    # inputs_hw = inputs.view(B, C_Kd_Kh_Kw//Kd, Kd, Dout, -1).permute(0,2,3,1,4).contiguous().view(B*Kd*Dout, C_Kd_Kh_Kw//Kd, -1) # B*Kd*Dout, C_Kd_Kh_Kw//Kd, Hout*Wout
    # inputs_hw_folded = F.fold(inputs_hw, (H, W), (Kh, Kw), dilation, padding, stride) # B*Kd*Dout, C, H, W
    # C = inputs_hw_folded.shape[1]
    # inputs_hw_folded = inputs_hw_folded.view(B, Kd, Dout, C, H, W, 1).permute(0,3,1,4,5,2,6).contiguous().view(B, C*Kd*H*W, Dout)
    # inputs_hwd_folded = F.fold(inputs_hw_folded, (D, 1), (Kd, 1), (dilation, 1), (padding, 0), (stride, 1)) # B, C*H*W, D, 1
    # inputs_hwd = inputs_hwd_folded.view(B, C, H, W, D).permute(0,1,4,2,3).contiguous()
    # return inputs_hwd

    B = inputs.shape[0]
    D, H, W = output_size
    Kd, Kh, Kw = kernel_size

    P = padding
    S = stride
    Dil = dilation

    Dout = (D + 2 * P - Dil * (Kd - 1) - 1) // S + 1
    Hout = (H + 2 * P - Dil * (Kh - 1) - 1) // S + 1
    Wout = (W + 2 * P - Dil * (Kw - 1) - 1) // S + 1

    Cin = inputs.shape[1] // (Kd * Kh * Kw)

    # [B, Cin*Kd*Kh*Kw, Dout*Hout*Wout]
    # -> [B, Cin, Kd, Kh, Kw, Dout, Hout, Wout]
    inputs = inputs.view(B, Cin, Kd, Kh, Kw, Dout, Hout, Wout)

    # Reverse the final permute from unfold3d:
    # [B, Cin, Kd, Kh, Kw, Dout, Hout, Wout]
    # -> [B, Cin, Kh, Kw, Hout, Wout, Kd, Dout]
    inputs = inputs.permute(0, 1, 3, 4, 6, 7, 2, 5).contiguous()

    # Prepare for depth fold:
    # [B, Cin, Kh, Kw, Hout, Wout, Kd, Dout]
    # -> [B, Cin*Kh*Kw*Hout*Wout*Kd, Dout]
    inputs_depth = inputs.view(B, Cin * Kh * Kw * Hout * Wout * Kd, Dout)

    # Fold depth:
    # [B, Cin*Kh*Kw*Hout*Wout*Kd, Dout]
    # -> [B, Cin*Kh*Kw*Hout*Wout, D, 1]
    inputs_depth_folded = F.fold(
        inputs_depth,
        output_size=(D, 1),
        kernel_size=(Kd, 1),
        dilation=(Dil, 1),
        padding=(P, 0),
        stride=(S, 1),
    )

    # Reverse preparation for depth unfold:
    # [B, Cin*Kh*Kw*Hout*Wout, D, 1]
    # -> [B, Cin, Kh, Kw, Hout, Wout, D]
    inputs_hw = inputs_depth_folded.view(B, Cin, Kh, Kw, Hout, Wout, D)

    # -> [B, D, Cin, Kh, Kw, Hout, Wout]
    inputs_hw = inputs_hw.permute(0, 6, 1, 2, 3, 4, 5).contiguous()

    # Prepare for spatial fold:
    # [B, D, Cin, Kh, Kw, Hout, Wout]
    # -> [B*D, Cin*Kh*Kw, Hout*Wout]
    inputs_hw = inputs_hw.view(B * D, Cin * Kh * Kw, Hout * Wout)

    # Fold height and width:
    # [B*D, Cin*Kh*Kw, Hout*Wout]
    # -> [B*D, Cin, H, W]
    outputs = F.fold(
        inputs_hw,
        output_size=(H, W),
        kernel_size=(Kh, Kw),
        dilation=Dil,
        padding=P,
        stride=S,
    )

    # [B*D, Cin, H, W]
    # -> [B, D, Cin, H, W]
    # -> [B, Cin, D, H, W]
    outputs = outputs.view(B, D, Cin, H, W).transpose(1, 2).contiguous()

    return outputs
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
        dilation: tuple[int, int, int],
        padding: tuple[int, int, int],
        stride: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This is the backward of the layer.
        """

        # TODO


class Conv3d(torch.nn.Module):
    """
    This is the class that represents the Conv3d layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int, int],
        dilation: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        stride: int | tuple[int, int, int] = 1,
    ) -> None:
        """
        This method is the constructor of the Conv3d layer.
        """

        super().__init__()
        kernel_size = _triple(kernel_size)
        self.dilation = _triple(dilation)
        self.padding = _triple(padding)
        self.stride = _triple(stride)

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
