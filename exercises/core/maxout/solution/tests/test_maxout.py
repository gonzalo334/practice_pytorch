""" """

# 3pps
import torch
import pytest

# Own modules
from src.maxout import MaxOut
from tests.utils import MaxOutLoop


@pytest.mark.parametrize("num_units, output_dim", [(2, 1), (10, 10)])
class TestMaxOut:
    def test_init(
        self, num_units: int, output_dim: int, inputs_linear: torch.Tensor
    ) -> None:
        """
        This method tests the init method of the MaxOut class.

        Args:
            num_units: number of units of the MaxOut.
            output_dim: Output dimension.
            inputs_linear: Inputs from the fixture.

        Returns:
            None.
        """

        # Define maxout layers
        maxout = MaxOut(num_units, inputs_linear.shape[1], output_dim)

        # Check number of units
        assert (
            maxout.num_units == num_units
        ), f"Incorrect number of units, expected {num_units} got {maxout.num_units}"

        # Check weight type
        assert isinstance(maxout.weight, torch.Tensor), (
            f"Incorrect type of weight, expected {torch.Tensor} got "
            f"{type(maxout.weight)}"
        )

        # Check weight shape
        shape: tuple[int, int, int] = (num_units, output_dim, inputs_linear.shape[1])
        assert (
            maxout.weight.shape == shape
        ), f"Incorrect weight shape, expected {shape} got {maxout.weight.shape}"

        # Check dtype
        assert maxout.weight.dtype == torch.double, (
            f"Incorrect weight dtype, expected {torch.double} and got "
            f"{maxout.weight.dtype}"
        )

        return None

    def test_inputs_reshape(
        self, num_units: int, output_dim: int, inputs_linear: torch.Tensor
    ) -> None:
        """
        This method tests the init method of the MaxOut class.

        Args:
            num_units: number of units of the MaxOut.
            output_dim: Output dimension.
            inputs_linear: Inputs from the fixture.

        Returns:
            None.
        """

        # Define maxout layers
        maxout = MaxOut(num_units, inputs_linear.shape[1], output_dim)

        # Reshape
        inputs_reshaped: torch.Tensor = maxout.reshape_inputs(inputs_linear)

        # Check shape
        shape: tuple[int, int, int] = (
            inputs_linear.shape[0] * num_units,
            1,
            inputs_linear.shape[-1],
        )
        assert (
            inputs_reshaped.shape == shape
        ), f"Incorrect shape, expected {shape} and got {inputs_reshaped.shape}"

        return None

    def test_forward(
        self, num_units: int, output_dim: int, inputs_linear: torch.Tensor
    ) -> None:
        """
        This function tests the MaxOut

        Args:
            num_units: number of units of the MaxOut.
            output_dim: Output dimension.
            inputs_linear: Inputs from the fixture.

        Returns:
            None.
        """

        # Define maxout layers
        maxout = MaxOut(num_units, inputs_linear.shape[1], output_dim)
        maxout_loop = MaxOutLoop(num_units, inputs_linear.shape[1], output_dim)

        # Compute
        maxout_loop.load_parameters(maxout)

        # Compute outputs
        outputs: torch.Tensor = maxout(inputs_linear)
        outputs_loop: torch.Tensor = maxout_loop(inputs_linear)

        # Check shape
        assert outputs.shape == outputs_loop.shape, (
            f"Incorrect outputs shape, expected {outputs_loop.shape} and got "
            f"{outputs.shape}"
        )

        return None
