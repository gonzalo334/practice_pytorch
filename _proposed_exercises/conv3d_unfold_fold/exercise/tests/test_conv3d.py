# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.conv3d import Conv3d, fold3d, unfold3d
from src.utils import parameters_to_double, set_seed


set_seed(42)


@pytest.mark.order(1)
def test_unfold3d() -> None:
    """
    This function is the test for the unfold3d operation.
    """

    inputs = torch.arange(2 * 3 * 4 * 5 * 6).view(2, 3, 4, 5, 6).double()
    inputs_unfolded = unfold3d(inputs, (2, 3, 2), stride=2)

    assert inputs_unfolded.shape == (2, 36, 12), "Incorrect shape of unfold"
    assert torch.equal(
        inputs[0, :, :2, :3, :2].reshape(-1), inputs_unfolded[0, :, 0]
    ), "Incorrect values of unfold"
    assert torch.equal(
        inputs[0, :, :2, :3, 2:4].reshape(-1), inputs_unfolded[0, :, 1]
    ), "Incorrect values of unfold"


@pytest.mark.order(2)
def test_fold3d() -> None:
    """
    This function is the test for the fold3d operation.
    """

    inputs = torch.rand(4, 3, 5, 6, 7).double()
    kernel_size = (2, 3, 2)
    stride = 2
    padding = 1

    inputs_folded = fold3d(
        unfold3d(inputs, kernel_size, padding=padding, stride=stride),
        inputs.shape[2:],
        kernel_size,
        padding=padding,
        stride=stride,
    )
    divisor = fold3d(
        unfold3d(torch.ones_like(inputs), kernel_size, padding=padding, stride=stride),
        inputs.shape[2:],
        kernel_size,
        padding=padding,
        stride=stride,
    )
    check_tensor = divisor * inputs

    assert inputs_folded.shape == inputs.shape, "Incorrect shape of fold"
    assert torch.equal(inputs_folded, check_tensor), "Incorrect values of fold"


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    [
        ((2, 3, 2), 1, 0, 1),
        ((2, 2, 3), 2, 1, 1),
        ((2, 2, 2), 1, 1, 2),
    ],
)
def test_conv3d_forward(
    kernel_size: tuple[int, int, int],
    stride: int,
    padding: int,
    dilation: int,
) -> None:
    """
    This function is the test for the Conv3d forward pass.
    """

    inputs = torch.rand(3, 4, 6, 7, 8).double()

    set_seed(42)
    model = Conv3d(4, 5, kernel_size, stride=stride, padding=padding, dilation=dilation)
    parameters_to_double(model)

    set_seed(42)
    model_torch = torch.nn.Conv3d(
        4,
        5,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    parameters_to_double(model_torch)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs)

    assert outputs.shape == outputs_torch.shape, "Incorrect output shape"
    assert torch.allclose(outputs, outputs_torch, atol=1e-6), "Incorrect forward"


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    [
        ((2, 3, 2), 1, 0, 1),
        ((2, 2, 3), 2, 1, 1),
        ((2, 2, 2), 1, 1, 2),
    ],
)
def test_conv3d_backward(
    kernel_size: tuple[int, int, int],
    stride: int,
    padding: int,
    dilation: int,
) -> None:
    """
    This function is the test for the Conv3d backward pass.
    """

    inputs = torch.rand(3, 4, 6, 7, 8).double()
    inputs.requires_grad_(True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    set_seed(42)
    model = Conv3d(4, 5, kernel_size, stride=stride, padding=padding, dilation=dilation)
    parameters_to_double(model)

    set_seed(42)
    model_torch = torch.nn.Conv3d(
        4,
        5,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    parameters_to_double(model_torch)

    outputs = model(inputs)
    outputs_torch = model_torch(inputs_torch)
    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert inputs.grad is not None
    assert model.weight.grad is not None
    assert model.bias.grad is not None
    assert model_torch.weight.grad is not None
    assert model_torch.bias.grad is not None
    assert inputs_torch.grad is not None
    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)
    assert torch.allclose(model.bias.grad, model_torch.bias.grad, atol=1e-6)
