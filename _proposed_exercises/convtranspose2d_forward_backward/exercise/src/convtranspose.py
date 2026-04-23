# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math
from typing import Any


class ConvTranspose2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ConvTranspose2d layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        output_padding: tuple[int, int],
        groups: int,
        dilation: tuple[int, int],
    ) -> torch.Tensor:
        """
        This is the forward method of the ConvTranspose2d layer.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, channels, height, width].
            weight: weights tensor. Dimensions:
                [in channels, out channels // groups, kernel height, kernel width].
            bias: bias tensor. Dimensions: [out channels].

        Returns:
            outputs tensor. Dimensions: [batch, out channels, out height, out width].
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        """
        This method is the backward of the ConvTranspose2d layer.
        """

        # TODO


class ConvTranspose2d(torch.nn.Module):
    """
    This is the class that represents the ConvTranspose2d layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        This method is the constructor of the ConvTranspose2d layer.
        """

        super().__init__()
        kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.output_padding = self._pair(output_padding)
        self.groups = groups
        self.dilation = self._pair(dilation)

        self.weight = torch.nn.Parameter(
            torch.empty(
                in_channels,
                out_channels // groups,
                kernel_size[0],
                kernel_size[1],
                dtype=dtype,
            )
        )
        self.bias = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype))

        self.reset_parameters()
        self.fn = ConvTranspose2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.
        """

        return self.fn(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
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

