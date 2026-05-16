# deep learning libraries
import torch

# other libraries
import inspect
import pytest

# own modules
import src.conv3d_loops as conv3d_loops_module
from src.conv3d_loops import Conv3d, _conv_output_size
from src.utils import parameters_to_double, set_seed


set_seed(42)


@pytest.mark.order(1)
def test_conv_output_size() -> None:
    """
    This function checks the output size helper.
    """

    output_size = _conv_output_size(
        (6, 7, 8),
        (2, 3, 2),
        dilation=(1, 2, 1),
        padding=(1, 2, 0),
        stride=(2, 1, 2),
    )

    assert output_size == (4, 7, 4)


@pytest.mark.order(2)
def test_no_builtin_convolution() -> None:
    """
    This function checks that the implementation does not call built-in
    convolution or unfold/fold helpers.
    """

    source = inspect.getsource(conv3d_loops_module)
    forbidden = [
        "torch.conv3d",
        "torch.nn.Conv3d",
        "nn.Conv3d",
        "F.conv3d",
        "functional.conv3d",
        "unfold(",
        "fold(",
    ]

    for token in forbidden:
        assert token not in source, f"Forbidden operation: {token}"


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    [
        ((2, 2, 2), 1, 0, 1),
        ((2, 3, 2), (2, 1, 2), (1, 1, 0), 1),
        ((2, 2, 2), 1, (1, 1, 1), (2, 1, 1)),
    ],
)
def test_conv3d_forward(
    kernel_size: tuple[int, int, int],
    stride: int | tuple[int, int, int],
    padding: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int],
) -> None:
    """
    This function is the test for the Conv3d forward pass.
    """

    inputs = torch.rand(2, 3, 4, 5, 6).double()

    set_seed(42)
    model = Conv3d(3, 4, kernel_size, stride=stride, padding=padding, dilation=dilation)
    parameters_to_double(model)

    set_seed(42)
    model_torch = torch.nn.Conv3d(
        3,
        4,
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
        ((2, 2, 2), 1, 0, 1),
        ((2, 3, 2), (2, 1, 2), (1, 1, 0), 1),
        ((2, 2, 2), 1, (1, 1, 1), (2, 1, 1)),
    ],
)
def test_conv3d_backward(
    kernel_size: tuple[int, int, int],
    stride: int | tuple[int, int, int],
    padding: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int],
) -> None:
    """
    This function is the test for the Conv3d backward pass.
    """

    inputs = torch.rand(2, 3, 4, 5, 6).double()
    inputs.requires_grad_(True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    set_seed(42)
    model = Conv3d(3, 4, kernel_size, stride=stride, padding=padding, dilation=dilation)
    parameters_to_double(model)

    set_seed(42)
    model_torch = torch.nn.Conv3d(
        3,
        4,
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
    assert inputs_torch.grad is not None
    assert model_torch.weight.grad is not None
    assert model_torch.bias.grad is not None
    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(model.weight.grad, model_torch.weight.grad, atol=1e-6)
    assert torch.allclose(model.bias.grad, model_torch.bias.grad, atol=1e-6)
