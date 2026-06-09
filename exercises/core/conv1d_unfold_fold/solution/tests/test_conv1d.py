# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.conv1d import unfold1d, fold1d, Conv1d
from src.utils import parameters_to_double, set_seed

# set seed
set_seed(42)


@pytest.mark.order(1)
def test_unfold1d() -> None:
    """
    This function is the test for the unfold1d.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 32, 100)

    # unfold inputs
    inputs_unfolded: torch.Tensor = unfold1d(inputs, 7)

    # check dimensions
    assert inputs_unfolded.shape == (64, 224, 94), "Incorrect shape of unfold"

    # check values
    assert (
        inputs[0, :, :7].reshape(-1) != inputs_unfolded[0, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, :, 1:8].reshape(-1) != inputs_unfolded[0, :, 1]
    ).sum().item() == 0, "Incorrect values of unfold"

    return None


@pytest.mark.order(2)
def test_fold1d() -> None:
    """
    This function is the test for the fold1d.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 32, 100).double()

    # compute fold version
    inputs_folded: torch.Tensor = fold1d(unfold1d(inputs, 7), 100, 7)

    # compute check tensor
    input_ones = torch.ones(inputs.shape, dtype=inputs.dtype)
    divisor = fold1d(unfold1d(input_ones, 7), 100, 7)
    check_tensor: torch.Tensor = divisor * inputs

    # check dimensions
    assert inputs_folded.shape == (64, 32, 100), "Incorrect shape of fold"

    # check values
    assert (inputs_folded != check_tensor).sum().item() == 0, "Incorrect values of fold"

    return None


@pytest.mark.order(3)
def test_conv1d_forward() -> None:
    """
    This function is the test for the conv1d forward.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 32, 100).double()

    # define our conv
    set_seed(42)
    model: torch.nn.Module = Conv1d(32, 16, 7)
    parameters_to_double(model)

    # compute outputs
    outputs = model(inputs)

    # define torch conv
    set_seed(42)
    model_torch: torch.nn.Module = torch.nn.Conv1d(32, 16, 7)
    parameters_to_double(model_torch)

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check values of the forward
    assert (
        outputs.round(decimals=4) != outputs_torch.round(decimals=4)
    ).sum().item() == 0, "Incorrect forward"

    return None


@pytest.mark.order(4)
def test_conv1d_backward() -> None:
    """
    This function is the test for the conv1d backwrad.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 32, 100).double()
    inputs.requires_grad_(True)

    # define conv
    set_seed(42)
    model: torch.nn.Module = Conv1d(32, 16, 7)
    parameters_to_double(model)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_inputs: torch.Tensor = inputs.grad.clone()
    grad_weight: torch.Tensor = model.weight.grad.clone()
    grad_bias: torch.Tensor = model.bias.grad.clone()

    # define conv
    set_seed(42)
    model_torch: torch.nn.Module = torch.nn.Conv1d(32, 16, 7)
    parameters_to_double(model_torch)

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads
    if (
        model_torch.weight.grad is None
        or model_torch.bias.grad is None
        or inputs.grad is None
    ):
        assert False, "Gradients not returned, none value detected"
    grad_inputs_torch: torch.Tensor = inputs.grad.clone()
    grad_weight_torch: torch.Tensor = model_torch.weight.grad.clone()
    grad_bias_torch: torch.Tensor = model_torch.bias.grad.clone()

    # check values of the inputs gradients
    assert (
        grad_inputs != grad_inputs_torch
    ).sum().item() == 0, "Incorrect inputs gradients"

    # check values of the weights gradients
    assert (
        grad_weight != grad_weight_torch
    ).sum().item() == 0, "Incorrect weights gradient"

    # check values of the bias gradients
    assert (grad_bias != grad_bias_torch).sum().item() == 0, "Incorrect bias gradients"

    return None
