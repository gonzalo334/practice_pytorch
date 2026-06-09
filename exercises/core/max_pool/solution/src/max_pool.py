# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional, Any


def unfold_max_pool_2d(
    inputs: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    """
    This function computes the unfold needed for the MaxPool2d.
    Since the maxpool only computes the max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. For that, we will
    include the channels into another dimension that will
    not be affected by the consequently operations.

    Args:
        inputs: inputs tensor. Dimensions: [batch, channels, height,
            width].
        kernel_size: size of the kernel to use. In this case the
            kernel will be symmetric, that is why only an integer is
            accepted.
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
    """

    # shrink channel dim
    inputs_unfolded: torch.Tensor = inputs.flatten(0, 1).unsqueeze(1)

    # unfold
    inputs_unfolded = F.unfold(
        inputs_unfolded, kernel_size, stride=stride, padding=padding
    )

    return inputs_unfolded


def fold_max_pool_2d(
    inputs: torch.Tensor,
    output_size: int,
    batch_size: int,
    kernel_size,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    This function computes the fold needed for the MaxPool2d.
    Since the maxpool only comute sthe max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. To do that, we
    this fold version recovers the channel dimensions before executing
    the fold operation.

    Args:
        inputs: inputs unfolded. Dimensions: [batch * channels,
            kernel size * kernel size, number of windows].
        output_size: output size for the fold, i.e., the height and
            the width.
        batch_size: batch size
        stride: stride to use in the maxpool operation. As in the case
            of the kernel size, the stride willm be symmetric.
        padding: padding to use in the maxpool operation. As in the
            case of the kernel.

    Returns:
        inputs folded. Dimensions: [batch, channels, height, width].
    """

    # unshrink channel dim
    inputs_folded: torch.Tensor = inputs.contiguous().view(
        batch_size, -1, inputs.shape[2]
    )

    # fold
    inputs_folded = F.fold(
        inputs_folded,
        output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    return inputs_folded


class MaxPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the MaxPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the MaxPool2d.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                channels, height, width].

        Returns:
            output of the layer. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # unfold inputs and compute outputs
        inputs_unfolded: torch.Tensor = unfold_max_pool_2d(
            inputs, kernel_size, stride, padding
        )

        # compute max positions
        outputs_unfolded: torch.Tensor
        outputs_unfolded, max_indexes = torch.max(inputs_unfolded, dim=1, keepdim=True)
        max_indexes = max_indexes[:, 0, :]

        # compute fold version
        output_size: int = inputs.shape[2] - kernel_size + 1
        outputs: torch.Tensor = fold_max_pool_2d(
            outputs_unfolded, output_size, inputs.shape[0], 1, stride, padding
        )

        # save elements for the backward
        ctx.save_for_backward(
            inputs_unfolded,
            max_indexes,
            torch.tensor(kernel_size),
            torch.tensor(stride),
            torch.tensor(padding),
        )

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        This method is the backward of the MaxPool2d.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            inputs gradients dimensions: [batch, channels,
                height, width].
            None value.
            None value.
            None value.
        """

        # load tensors from the forward
        (
            inputs_unfolded,
            max_indexes,
            kernel_size_tensor,
            stride_tensor,
            padding_tensor,
        ) = ctx.saved_tensors

        # pass attributes to int
        kernel_size: int = kernel_size_tensor.item()
        stride: int = stride_tensor.item()
        padding: int = padding_tensor.item()

        # compute unfold of grad_outputs
        grad_outputs_unfolded: torch.Tensor = unfold_max_pool_2d(
            grad_outputs, 1, stride, padding
        )

        # compute grad inputs
        max_indexes = F.one_hot(max_indexes, num_classes=kernel_size**2)
        max_indexes = max_indexes.permute(0, 2, 1)
        grad_inputs_unfolded: torch.Tensor = max_indexes * grad_outputs_unfolded

        # fold grad inputs
        output_size: int = grad_outputs.shape[2] + kernel_size - 1
        grad_inputs: torch.Tensor = fold_max_pool_2d(
            grad_inputs_unfolded,
            output_size,
            grad_outputs.shape[0],
            kernel_size,
            stride,
            padding,
        )

        return grad_inputs, None, None, None


class MaxPool2d(torch.nn.Module):
    """
    This is the class that represents the MaxPool2d Layer.
    """

    kernel_size: int
    stride: int

    def __init__(
        self, kernel_size: int, stride: Optional[int], padding: int = 0
    ) -> None:
        """
        This method is the constructor of the MaxPool2d layer.
        """

        # call super class constructor
        super().__init__()

        # set attributes value
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

        # save function
        self.fn = MaxPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.kernel_size, self.stride, self.padding)
