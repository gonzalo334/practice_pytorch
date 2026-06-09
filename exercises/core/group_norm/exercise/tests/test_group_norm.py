# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.group_norm import GroupNorm
from src.utils import set_seed


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "shape, num_groups, affine",
    [((64, 6, 32, 32), 2, False), ((128, 8, 1024), 4, True)],
)
def test_group_norm(shape: tuple[int, ...], num_groups: int, affine: bool) -> None:
    """
    This is the test for the forward of the GroupNorm.

    Args:
        shape: shape of the input tensor.
        num_groups: number of groups in the input tensor.
        affine: bool to indicate affine transformation.

    """

    # loop with different seeds
    for seed in range(10):
        # define inputs
        scale = torch.randint(1, 10, (1,)).double()
        bias = torch.randint(-10, 10, (1,)).double()
        inputs: torch.Tensor = torch.rand(shape, dtype=torch.double) * scale + bias

        # define models
        set_seed(seed)
        model = GroupNorm(
            num_groups, shape[1], affine=affine, eps=0, dtype=torch.double
        )
        set_seed(seed)
        model_torch = torch.nn.GroupNorm(
            num_groups, shape[1], affine=affine, eps=0, dtype=torch.double
        )

        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-3), "Incorrect outputs"

    return None
