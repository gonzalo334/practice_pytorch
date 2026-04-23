# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.utils import set_seed
from src.models import Residual


# set seed and device
set_seed(42)


@pytest.mark.order(1)
def test_residual_forward() -> None:
    """
    This function is the test for the forward of the residual function.
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 3).uniform_(-10, 10)
    inputs[0, 0] = 0

    # define implemented prelu
    set_seed(42)
    model: torch.nn.Module = Residual(3)

    # compute outputs and backward
    outputs = model(inputs)

    # create torch class residual
    class ResidualTorch(torch.nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()

            self.linear = torch.nn.Linear(input_dim, input_dim)
            self.prelu = torch.nn.PReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.prelu(self.linear(inputs)) + inputs

    # define torch residual
    set_seed(42)
    model_torch: torch.nn.Module = ResidualTorch(3)

    # compute outputs and backward
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs.round(decimals=4) != outputs_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect forward"

    return None


@pytest.mark.order(2)
def test_residual_backward() -> None:
    """
    This function is the test for the backward of the residual function.
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 3).uniform_(-10, 10)
    inputs[0, 0] = 0
    inputs.requires_grad_(True)

    # define implemented prelu
    set_seed(42)
    model: torch.nn.Module = Residual(3)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_weight: torch.Tensor = model.weight.grad.clone()
    grad_bias: torch.Tensor = model.bias.grad.clone()
    grad_a: torch.Tensor = model.a.grad.clone()
    inputs_grad: torch.Tensor = inputs.grad.clone()

    # create torch class residual
    class ResidualTorch(torch.nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()

            self.linear = torch.nn.Linear(input_dim, input_dim)
            self.prelu = torch.nn.PReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.prelu(self.linear(inputs)) + inputs

    # define torch residual
    set_seed(42)
    model_torch: torch.nn.Module = ResidualTorch(3)

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    model_torch.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads values
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_weight_torch: torch.Tensor = model_torch.linear.weight.grad.clone()
    grad_bias_torch: torch.Tensor = model_torch.linear.bias.grad.clone()
    grad_a_torch: torch.Tensor = model_torch.prelu.weight.grad.clone()
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()

    # check weights grads
    assert (
        grad_weight.round(decimals=4) != grad_weight_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect weights gradients"

    # check bias grads
    assert (
        grad_bias.round(decimals=4) != grad_bias_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect bias gradients"

    # check a parameter
    assert (
        grad_a.round(decimals=4) != grad_a_torch[0].round(decimals=4)
    ).sum().item() == 0, "Incorrect 'a' parameter gradients"

    # check inputs grads
    assert (
        inputs_grad.round(decimals=4) != inputs_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect inputs gradients"

    return None
