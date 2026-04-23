# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.softshrink import Softshrink
from src.utils import set_seed


@pytest.mark.order(1)
@pytest.mark.parametrize("shape, lambd", [((64, 32), 0.5), ((64, 3, 32, 32), 1.0)])
def test_softshrink_forward(shape: tuple[int, ...], lambd: float) -> None:
    """
    This function is the forward test for the Softshrink.

    Args:
        shape: shape of the tensors to use.
        lambd: lambd parameter to use.
    """

    for seed in range(10):
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(shape)

        # put elements on the frontier
        inputs[0, 0] = 0.5
        inputs[0, 1] = -0.5

        # define models
        model = Softshrink(lambd)
        model_torch = torch.nn.Softshrink(lambd)

        # compute outputs
        outputs: torch.Tensor = model(inputs)
        outputs_torch: torch.Tensor = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-4), "Incorrect outputs"

    return None


@pytest.mark.order(2)
@pytest.mark.parametrize("shape, lambd", [((64, 32), 0.5), ((64, 3, 32, 32), 1.0)])
def test_softshrink_backward(shape: tuple[int, ...], lambd: float) -> None:
    """
    This function is the backward test for the Softshrink.

    Args:
        shape: shape of the tensors to use.
        lambd: lambd parameter to use.
    """

    for seed in range(10):
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(shape)

        # put elements on the frontier
        inputs[0, 0] = 0.5
        inputs[0, 1] = -0.5

        inputs.requires_grad_(True)

        # define models
        model = Softshrink(lambd)
        model_torch = torch.nn.Softshrink(lambd)

        # compute backward of our maxpool
        outputs = model(inputs)
        if inputs.grad is not None:
            inputs.grad.zero_()
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()

        # compute backward of pytorch maxpool
        outputs_torch = model_torch(inputs)
        inputs.grad.zero_()
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch: torch.Tensor = inputs.grad.clone()

        # check output size
        assert (
            grad_inputs.shape == grad_inputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(
            grad_inputs, grad_inputs_torch, atol=1e-4
        ), "Incorrect outputs"

    return None
