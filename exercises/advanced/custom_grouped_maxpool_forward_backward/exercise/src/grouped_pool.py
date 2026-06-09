# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Any


class GroupedMaxPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of a grouped max-pooling operation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        num_groups: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> torch.Tensor:
        """
        This is the forward method of grouped max-pooling.
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """
        This method is the backward of grouped max-pooling.
        """

        # TODO


class GroupedMaxPool2d(torch.nn.Module):
    """
    This is the class that represents the grouped max-pooling layer.
    """

    def __init__(
        self,
        num_groups: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride if stride is not None else kernel_size)
        self.padding = self._pair(padding)
        self.fn = GroupedMaxPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fn(
            inputs,
            self.num_groups,
            self.kernel_size,
            self.stride,
            self.padding,
        )

    @staticmethod
    def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, tuple):
            return value
        return (value, value)

