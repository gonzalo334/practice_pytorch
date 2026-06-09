"""
This module contains the code to test the ModePool.
"""

# 3pps
import torch
import pytest

# Own modules
from src.pool import CustomMaxPool2d
from tests.utils import TestCustomMaxPool2d


@pytest.mark.parametrize("kernel_size, num_groups", [(3, 2), (4, 3)])
class TestMaxPool2d:
    """
    This is the class to test the CustomMaxPool2d.
    """

    @torch.no_grad()
    def test_forward(
        self, kernel_size: int, num_groups: int, inputs_2d: torch.Tensor
    ) -> None:
        """
        This function is the test of the forward pass.

        Args:
            kernel_size: Kernel size.
            num_groups: Number of groups.
            inputs_2d: Inputs fixture. Dimensions: [batch size,
                number of channels, height, width].

        Returns:
            None.
        """

        # Rename fixture
        inputs: torch.Tensor = inputs_2d

        # Define modules
        module: torch.nn.Module = CustomMaxPool2d(kernel_size, num_groups)
        module_test: torch.nn.Module = TestCustomMaxPool2d(kernel_size, num_groups)

        # Compute outputs
        outputs: torch.Tensor = module(inputs)
        outputs_test: torch.Tensor = module_test(inputs)

        # Check shape
        assert (
            outputs.shape == outputs_test.shape
        ), f"Incorrect shape, expected {outputs_test.shape} and got {outputs.shape}"

        # Check values
        assert torch.allclose(outputs, outputs_test), "Incorrect values in forward pass"

        return None

    @pytest.mark.parametrize("use_grad_outputs", [False, True])
    def test_backward(
        self,
        kernel_size: int,
        num_groups: int,
        use_grad_outputs: bool,
        inputs_2d: torch.Tensor,
    ) -> None:
        """
        This method is the test for the backward method.

        Args:
            kernel_size: Kernel size.
            num_groups: Number of groups.
            use_grad_outputs: Indicator to use grad outputs.
            inputs_2d: Inputs fixture. Dimensions: [batch size,
                number of channels, height, width].

        Raises:
            RuntimeError: Error in gradient computation.

        Returns:
            None.
        """

        # Rename fixture
        inputs: torch.Tensor = inputs_2d
        inputs.requires_grad_(True)

        # Define modules
        module: torch.nn.Module = CustomMaxPool2d(kernel_size, num_groups)
        module_test: torch.nn.Module = TestCustomMaxPool2d(kernel_size, num_groups)

        # Compute outputs and weights
        outputs: torch.Tensor = module(inputs)
        weights: torch.Tensor
        if use_grad_outputs:
            weights = torch.rand_like(outputs)
        else:
            weights = torch.ones_like(outputs)

        # Compute gradients
        (outputs * weights).sum().backward()
        if inputs.grad is None:
            raise RuntimeError("Error in gradient computation")
        gradients: torch.Tensor = inputs.grad.clone()
        inputs.grad.zero_()

        # Compute torch gradients
        outputs = module_test(inputs)
        (outputs * weights).sum().backward()
        if inputs.grad is None:
            raise RuntimeError("Error in gradient computation")
        gradients_test: torch.Tensor = inputs.grad.clone()
        inputs.grad.zero_()

        # Check values
        assert torch.allclose(gradients, gradients_test)

        return None
