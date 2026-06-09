# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class EmbeddingRNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of a tanh RNN
    with embedding inputs.
    """

    @staticmethod
    def forward(
        ctx: Any,
        token_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_h: torch.Tensor,
        truncate_steps: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the RNN.
        """

        batch, seq_len = token_ids.shape
        hidden_dim = weight_hh.shape[0]
        hidden = torch.zeros(batch, hidden_dim, dtype=embedding_weight.dtype)
        hidden_states = []
        input_states = []

        for time_step in range(seq_len):
            inputs = embedding_weight[token_ids[:, time_step]]
            hidden = torch.tanh(inputs @ weight_ih.T + hidden @ weight_hh.T + bias_h)
            input_states.append(inputs)
            hidden_states.append(hidden)

        outputs = torch.stack(hidden_states, dim=1)
        inputs_stacked = torch.stack(input_states, dim=1)

        ctx.save_for_backward(
            token_ids,
            inputs_stacked,
            outputs,
            embedding_weight,
            weight_ih,
            weight_hh,
        )
        ctx.truncate_steps = truncate_steps

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        This method is the backward of the RNN.
        """

        (
            token_ids,
            input_states,
            hidden_states,
            embedding_weight,
            weight_ih,
            weight_hh,
        ) = ctx.saved_tensors

        batch, seq_len = token_ids.shape
        hidden_dim = weight_hh.shape[0]
        grad_embedding_weight = torch.zeros_like(embedding_weight)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_h = torch.zeros(hidden_dim, dtype=grad_outputs.dtype)
        grad_hidden_next = torch.zeros(batch, hidden_dim, dtype=grad_outputs.dtype)

        for time_step in range(seq_len - 1, -1, -1):
            grad_hidden = grad_outputs[:, time_step, :] + grad_hidden_next
            hidden = hidden_states[:, time_step, :]
            grad_pre_activation = grad_hidden * (1 - hidden**2)

            inputs = input_states[:, time_step, :]
            previous_hidden = (
                torch.zeros_like(hidden)
                if time_step == 0
                else hidden_states[:, time_step - 1, :]
            )

            grad_weight_ih += grad_pre_activation.T @ inputs
            grad_weight_hh += grad_pre_activation.T @ previous_hidden
            grad_bias_h += grad_pre_activation.sum(dim=0)

            grad_inputs = grad_pre_activation @ weight_ih
            grad_embedding_weight.index_add_(0, token_ids[:, time_step], grad_inputs)

            grad_hidden_previous = grad_pre_activation @ weight_hh
            if (
                ctx.truncate_steps > 0
                and time_step > 0
                and time_step % ctx.truncate_steps == 0
            ):
                grad_hidden_next = torch.zeros_like(grad_hidden_previous)
            else:
                grad_hidden_next = grad_hidden_previous

        return (
            None,
            grad_embedding_weight,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_h,
            None,
        )


class EmbeddingRNN(torch.nn.Module):
    """
    This is the class that represents the tanh RNN with embedding inputs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        truncate_steps: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.truncate_steps = truncate_steps
        self.embedding_weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        )
        self.weight_ih = torch.nn.Parameter(
            torch.empty(hidden_dim, embedding_dim, dtype=dtype)
        )
        self.weight_hh = torch.nn.Parameter(
            torch.empty(hidden_dim, hidden_dim, dtype=dtype)
        )
        self.bias_h = torch.nn.Parameter(torch.empty(hidden_dim, dtype=dtype))
        self.reset_parameters()
        self.fn = EmbeddingRNNFunction.apply

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.fn(
            token_ids,
            self.embedding_weight,
            self.weight_ih,
            self.weight_hh,
            self.bias_h,
            self.truncate_steps,
        )

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.embedding_weight, mean=0.0, std=0.1)
        bound = 1 / math.sqrt(self.weight_ih.shape[1])
        torch.nn.init.uniform_(self.weight_ih, -bound, bound)
        torch.nn.init.uniform_(self.weight_hh, -bound, bound)
        torch.nn.init.uniform_(self.bias_h, -bound, bound)

