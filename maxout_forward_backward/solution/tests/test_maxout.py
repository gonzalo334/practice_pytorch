# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.maxout import Maxout
from src.utils import parameters_to_double, set_seed

# set seed
set_seed(42)


@pytest.mark.order(5)
@torch.no_grad()
def test_maxout_forward() -> None:
    """
    This function test the maxout.
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 32).uniform_(-10, 10)
    inputs[0, 0] = 0

    # define maxout
    model = Maxout(32, 32)
    model.set_parameters(
        torch.zeros_like(model.weights_first),
        torch.zeros_like(model.bias_first),
        torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_second),
    )

    # compute outputs and backward
    outputs = model(inputs)

    # define torch relu
    model_torch: torch.nn.Module = torch.nn.ReLU()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward simulating the relu"

    # define maxout
    model = Maxout(32, 32)
    model.set_parameters(
        -1 * torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_first),
        torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_second),
    )

    # compute outputs and backward
    outputs = model(inputs)

    # compute outputs
    outputs_torch = torch.abs(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward simulating the absolute value"

    # define maxout
    model = Maxout(32, 20)

    # compute outputs and backward
    outputs = model(inputs)

    # define torch relu
    class MaxoutTorch(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            # call super class constructor
            super().__init__()

            self.linear1 = torch.nn.Linear(input_dim, output_dim)
            self.linear2 = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs1: torch.Tensor = self.linear1(inputs)
            outputs2: torch.Tensor = self.linear2(inputs)
            outputs = torch.maximum(outputs1, outputs2)

            return outputs

    # compute outputs
    model_torch = MaxoutTorch(32, 20)
    model_torch.linear1.weight.data = model.weights_first.clone()
    model_torch.linear1.bias.data = model.bias_first.clone()
    model_torch.linear2.weight.data = model.weights_second.clone()
    model_torch.linear2.bias.data = model.bias_second.clone()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward of implementation with nn"

    return None


@pytest.mark.order(6)
def test_maxout_backward() -> None:
    """
    _summary_

    Returns:
        _description_
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 32).uniform_(-10, 10)
    inputs[0, 0] = 0
    inputs.requires_grad_(True)

    # define maxout
    model = Maxout(32, 20)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if (
        inputs.grad is None
        or model.weights_first.grad is None
        or model.bias_first.grad is None
        or model.weights_second.grad is None
        or model.bias_second.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    inputs_grad: torch.Tensor = inputs.grad.clone()
    grad_weight_first: torch.Tensor = model.weights_first.grad.clone()
    grad_bias_first: torch.Tensor = model.bias_first.grad.clone()
    grad_weight_second: torch.Tensor = model.weights_second.grad.clone()
    grad_bias_second: torch.Tensor = model.bias_second.grad.clone()

    # define torch relu
    class MaxoutTorch(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            # call super class constructor
            super().__init__()

            self.linear1 = torch.nn.Linear(input_dim, output_dim)
            self.linear2 = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs1: torch.Tensor = self.linear1(inputs)
            outputs2: torch.Tensor = self.linear2(inputs)
            outputs = torch.maximum(outputs1, outputs2)

            return outputs

    # compute outputs
    model_torch = MaxoutTorch(32, 20)
    with torch.no_grad():
        model_torch.linear1.weight.data = model.weights_first.clone()
        model_torch.linear1.bias.data = model.bias_first.clone()
        model_torch.linear2.weight.data = model.weights_second.clone()
        model_torch.linear2.bias.data = model.bias_second.clone()

    # compute outputs
    outputs_torch = model_torch(inputs)
    model.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if (
        inputs.grad is None
        or model_torch.linear1.weight.grad is None
        or model_torch.linear1.bias.grad is None
        or model_torch.linear2.weight.grad is None
        or model_torch.linear2.bias.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()
    grad_weight_first_torch: torch.Tensor = model_torch.linear1.weight.grad.clone()
    grad_bias_first_torch: torch.Tensor = model_torch.linear1.bias.grad.clone()
    grad_weight_second_torch: torch.Tensor = model_torch.linear2.weight.grad.clone()
    grad_bias_second_torch: torch.Tensor = model_torch.linear2.bias.grad.clone()

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect inputs gradients"

    # check weights first grads
    assert (
        grad_weight_first != grad_weight_first_torch
    ).sum().item() == 0, "Incorrect first weights gradients"

    # check bias second grads
    assert (
        grad_bias_first != grad_bias_first_torch
    ).sum().item() == 0, "Incorrect first bias gradients"

    # check weights first grads
    assert (
        grad_weight_second != grad_weight_second_torch
    ).sum().item() == 0, "Incorrect second weights gradients"

    # check bias second grads
    assert (
        grad_bias_second != grad_bias_second_torch
    ).sum().item() == 0, "Incorrect second bias gradients"

    return None
