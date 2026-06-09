import torch
import torch.nn.functional as F
import pytest

from src.grouped_pool import GroupedMaxPool2d


def grouped_pool_reference(
    inputs: torch.Tensor,
    num_groups: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    batch, channels, height, width = inputs.shape
    channels_per_group = channels // num_groups
    inputs_grouped = inputs.view(batch, num_groups, channels_per_group, height, width)
    outputs = F.max_pool3d(
        inputs_grouped,
        kernel_size=(channels_per_group, kernel_size[0], kernel_size[1]),
        stride=(channels_per_group, stride[0], stride[1]),
        padding=(0, padding[0], padding[1]),
    )
    return outputs.squeeze(2)


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "num_groups, kernel_size, stride, padding",
    [(2, (2, 2), (2, 1), (0, 0)), (3, (3, 2), (1, 2), (1, 0))],
)
def test_grouped_maxpool_forward_backward(
    num_groups: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 6, 6, 5, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)

    model = GroupedMaxPool2d(num_groups, kernel_size, stride=stride, padding=padding)
    outputs = model(inputs)
    outputs_torch = grouped_pool_reference(
        inputs_torch, num_groups, kernel_size, stride, padding
    )

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)

