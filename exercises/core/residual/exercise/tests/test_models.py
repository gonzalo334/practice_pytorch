# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.utils import set_seed
from src.models import PReLU, Residual


# set seed and device
set_seed(42)


@pytest.mark.order(2)
def test_prelu_forward() -> None:
    """
    This function is the test for the forward of the prelu function.
    """

    set_seed(42)

    inputs: torch.Tensor = torch.FloatTensor(64, 3).uniform_(-10, 10)
    inputs[0, 0] = 0

    model: torch.nn.Module = PReLU()
    outputs = model(inputs)

    model_torch: torch.nn.Module = torch.nn.PReLU()
    outputs_torch = model_torch(inputs)

    assert (
        outputs.round(decimals=4) != outputs_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect forward"

    return None


@pytest.mark.order(3)
def test_prelu_backward() -> None:
    """
    This function is the test for the backward of the prelu function.
    """

    set_seed(42)

    inputs: torch.Tensor = torch.FloatTensor(64, 3).uniform_(-10, 10)
    inputs[0, 0] = 0
    inputs.requires_grad_(True)

    model: torch.nn.Module = PReLU()

    outputs = model(inputs)
    outputs.sum().backward()

    if inputs.grad is None or model.a.grad is None:
        assert False, "Gradients not returned, none value detected"

    inputs_grad: torch.Tensor = inputs.grad.clone()
    a_grad: torch.Tensor = model.a.grad.clone()

    model_torch: torch.nn.Module = torch.nn.PReLU()

    outputs_torch = model_torch(inputs)
    model_torch.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    if inputs.grad is None or model_torch.weight.grad is None:
        assert False, "Gradients not returned, none value detected"

    inputs_grad_torch: torch.Tensor = inputs.grad.clone()
    a_grad_torch: torch.Tensor = model_torch.weight.grad.clone()

    assert (
        inputs_grad.round(decimals=4) != inputs_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect inputs gradients"

    assert (
        a_grad.round(decimals=4) != a_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect 'a' parameter gradients"

    return None


@pytest.mark.order(4)
def test_residual_forward() -> None:
    """
    This function is the test for the forward of the residual function.
    """

    set_seed(42)

    inputs: torch.Tensor = torch.FloatTensor(64, 3).uniform_(-10, 10)
    inputs[0, 0] = 0

    set_seed(42)
    model: torch.nn.Module = Residual(3)

    outputs = model(inputs)

    class ResidualTorch(torch.nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()

            self.linear = torch.nn.Linear(input_dim, input_dim)
            self.prelu = torch.nn.PReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.prelu(self.linear(inputs)) + inputs

    set_seed(42)
    model_torch: torch.nn.Module = ResidualTorch(3)

    outputs_torch = model_torch(inputs)

    assert (
        outputs.round(decimals=4) != outputs_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect forward"

    return None


@pytest.mark.order(5)
def test_residual_backward() -> None:
    """
    This function is the test for the backward of the residual function.
    """

    set_seed(42)

    inputs: torch.Tensor = torch.FloatTensor(64, 3).uniform_(-10, 10)
    inputs[0, 0] = 0
    inputs.requires_grad_(True)

    set_seed(42)
    model: torch.nn.Module = Residual(3)

    outputs = model(inputs)
    outputs.sum().backward()

    if (
        model.weight.grad is None
        or model.bias.grad is None
        or model.a.grad is None
        or inputs.grad is None
    ):
        assert False, "Gradients not returned, none value detected"

    grad_weight: torch.Tensor = model.weight.grad.clone()
    grad_bias: torch.Tensor = model.bias.grad.clone()
    grad_a: torch.Tensor = model.a.grad.clone()
    inputs_grad: torch.Tensor = inputs.grad.clone()

    class ResidualTorch(torch.nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()

            self.linear = torch.nn.Linear(input_dim, input_dim)
            self.prelu = torch.nn.PReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.prelu(self.linear(inputs)) + inputs

    set_seed(42)
    model_torch: torch.nn.Module = ResidualTorch(3)

    outputs_torch = model_torch(inputs)
    model_torch.zero_grad()
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    if (
        model_torch.linear.weight.grad is None
        or model_torch.linear.bias.grad is None
        or model_torch.prelu.weight.grad is None
        or inputs.grad is None
    ):
        assert False, "Gradients not returned, none value detected"

    grad_weight_torch: torch.Tensor = model_torch.linear.weight.grad.clone()
    grad_bias_torch: torch.Tensor = model_torch.linear.bias.grad.clone()
    grad_a_torch: torch.Tensor = model_torch.prelu.weight.grad.clone()
    inputs_grad_torch: torch.Tensor = inputs.grad.clone()

    assert (
        grad_weight.round(decimals=4) != grad_weight_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect weights gradients"

    assert (
        grad_bias.round(decimals=4) != grad_bias_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect bias gradients"

    assert (
        grad_a.round(decimals=4) != grad_a_torch[0].round(decimals=4)
    ).sum().item() == 0, "Incorrect 'a' parameter gradients"

    assert (
        inputs_grad.round(decimals=4) != inputs_grad_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect inputs gradients"

    return None