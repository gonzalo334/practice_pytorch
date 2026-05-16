# deep leanring libraries
import torch
import torch.nn.functional as F

# other libraries
import math
from typing import Any


def unfold1d(
    inputs: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """
    This operation computes the unfold operation for 1d convolutions.

    Args:
        inputs: input tensor. Dimensions: [batch, input channels,
            input sequence length].
        kernel_size: kernel size of the unfold operation.
        dilation: dilation of the unfold operation. Defaults to 1.
        padding: padding of the unfold operation. Defaults to 0.
        stride: stride of the unfold operation. Defaults to 1.

    Returns:
        outputs tensor. Dimensions: [batch,
            input channels * kernel size, number of windows].
    """

    # TODO
    inputs_unfolded = torch.nn.functional.unfold(inputs.unsqueeze(-1), (kernel_size, 1), dilation, padding, stride)
    return inputs_unfolded


def fold1d(
    inputs: torch.Tensor,
    output_size: int,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
) -> torch.Tensor:
    """
    This operation computes the fold operation for 1d convolutions.

    Args:
        inputs: input tensor. Dimensions: [batch,
            output channels * kernel size, number of windows].
        output_size: output sequence length.
        kernel_size: kernel size to use in the fold operation.
        dilation: dilation to use in the fold operation.
        stride: stride to use in the fold operation.

    Returns:
        output tensor. Dimensions: [batch, output channels,
            output sequence length].
    """

    # TODO
    inputs_folded = torch.nn.functional.fold(inputs, (output_size, 1), (kernel_size, 1), dilation, padding, stride)
    return inputs_folded.squeeze()
class Conv1dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv1d
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

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                input channels, sequence length].
            weight: weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size].
            bias: bias of the layer. Dimensions: [output channels].

        Returns:
            output of the layer. Dimensions:
                [batch, output channels,
                (sequence length + 2*padding - kernel size) /
                stride + 1]
        """

        # TODO
        B, Cin, L = inputs.shape
        Cout, Cin, K = weight.shape
        Lout = (L + 2*padding - K) // stride + 1
        inputs_unfolded = unfold1d(inputs, K, dilation, padding, stride) # B, Cin*K, Lout
        weight_unfolded = weight.view(Cout, Cin*K).unsqueeze(0) # 1, Cout, Cin*K
        bias_unfolded = bias.unsqueeze(0).unsqueeze(-1) # 1, Cout, 1
        outputs = weight_unfolded @ inputs_unfolded + bias_unfolded # B, Cout, Lout
        ctx.save_for_backward(inputs, weight)
        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation
        return outputs
        

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: contex for loading elements needed in the backward.
            grad_output: outputs gradients. Dimensions:
                [batch, output channels,
                (sequence length + 2*padding - kernel size) /
                stride + 1]

        Returns:
            gradient of the inputs. Dimensions: [batch,
                input channels, sequence length].
            gradient of the weights. Dimensions: [output channels,
                input channels, kernel size].
            gradient of the bias. Dimensions: [output channels].
            None.
            None.
            None.
        """

        # TODO
        inputs, weight = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride 
        dilation = ctx.dilation
        B, Cin, L = inputs.shape
        Cout, Cin, K = weight.shape
        B, Cout, Lout = grad_output.shape
        inputs_unfolded = unfold1d(inputs, K, dilation, padding, stride).transpose(1,2) # B, Lout, Cin*K
        weight_unfolded = weight.view(Cout, Cin*K).T.unsqueeze(0) # 1, Cin*K, Cout
        grad_inputs_unfolded = weight_unfolded @ grad_output # B, Cin*K, Lout
        grad_weights = (grad_output @ inputs_unfolded).sum(dim=0).view(Cout, Cin, K)
        grad_inputs = fold1d(grad_inputs_unfolded, L, K, dilation, padding, stride) # B, Cin, L
        grad_bias = grad_output.sum(dim=(0,2))
        return grad_inputs, grad_weights, grad_bias, None, None, None

class Conv1d(torch.nn.Module):
    """
    This is the class that represents the Conv1d Layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: input dimension.
            output_channels: output dimension.
            kernel_size: kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv1dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input channels,
                output channels, sequence length].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                sequence length - kernel size + 1].
        """

        return self.fn(
            inputs, self.weight, self.bias, self.dilation, self.padding, self.stride
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None
