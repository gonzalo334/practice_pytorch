# deep learning libraries
import torch

# other libraries
from typing import Any, Optional


class EmbeddingFuncion(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Embedding layer.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        padding_idx: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the Embedding layer.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # compute embeddings
        outputs: torch.Tensor = weight[inputs, :]

        # save tensors for the backward
        ctx.save_for_backward(inputs, weight, torch.tensor(padding_idx))

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        """
        This method is the backward of the Embedding layer.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, output dim].

        Returns:
            None value.
            inputs gradients. Dimensions: [batch].
            None value.
        """

        # load tensors from the forward
        (inputs, weight, padding_idx_tensor) = ctx.saved_tensors

        # pass attributes to int
        padding_idx: int = padding_idx_tensor.item()

        # init grad weights
        grad_weight: torch.Tensor = torch.zeros_like(weight)

        # fill grad weights
        for i in range(weight.shape[0]):
            if i != padding_idx:
                grad_weight[i, :] = torch.sum(grad_outputs[inputs == i], dim=0)

        return (None, grad_weight, None)


class Embedding(torch.nn.Module):
    """
    This is the class that represents the Embedding Layer.
    """

    padding_idx: int

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        """
        This method is the constructor of the Embedding layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        # set padding idx
        self.padding_idx = padding_idx if padding_idx is not None else -1

        self.fn = EmbeddingFuncion.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.padding_idx)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight)
