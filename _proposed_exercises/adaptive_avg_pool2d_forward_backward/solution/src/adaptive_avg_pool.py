# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class AdaptiveAvgPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of AdaptiveAvgPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Forward pass of AdaptiveAvgPool2d.
        """

        batch, channels, input_height, input_width = inputs.shape
        output_height, output_width = output_size
        outputs = torch.empty(
            batch,
            channels,
            output_height,
            output_width,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        for out_h in range(output_height):
            h_start = math.floor(out_h * input_height / output_height)
            h_end = math.ceil((out_h + 1) * input_height / output_height)
            for out_w in range(output_width):
                w_start = math.floor(out_w * input_width / output_width)
                w_end = math.ceil((out_w + 1) * input_width / output_width)
                outputs[:, :, out_h, out_w] = inputs[
                    :, :, h_start:h_end, w_start:w_end
                ].mean(dim=(2, 3))

        ctx.input_shape = inputs.shape
        ctx.output_size = output_size
        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        Backward pass of AdaptiveAvgPool2d.
        """

        batch, channels, input_height, input_width = ctx.input_shape
        output_height, output_width = ctx.output_size
        grad_inputs = torch.zeros(
            batch,
            channels,
            input_height,
            input_width,
            dtype=grad_outputs.dtype,
            device=grad_outputs.device,
        )

        for out_h in range(output_height):
            h_start = math.floor(out_h * input_height / output_height)
            h_end = math.ceil((out_h + 1) * input_height / output_height)
            for out_w in range(output_width):
                w_start = math.floor(out_w * input_width / output_width)
                w_end = math.ceil((out_w + 1) * input_width / output_width)
                area = (h_end - h_start) * (w_end - w_start)
                grad_inputs[:, :, h_start:h_end, w_start:w_end] += (
                    grad_outputs[:, :, out_h, out_w].unsqueeze(-1).unsqueeze(-1) / area
                )

        return grad_inputs, None


class AdaptiveAvgPool2d(torch.nn.Module):
    """
    This is the class that represents AdaptiveAvgPool2d.
    """

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size
        self.fn = AdaptiveAvgPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(inputs, self.output_size)
