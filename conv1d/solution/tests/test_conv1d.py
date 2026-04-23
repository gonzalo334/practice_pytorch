"""
This module contains the code to test the Conv1d.
"""

# 3pps
import torch
import pytest

# Own modules
from src.conv1d import Conv1d


@pytest.mark.parametrize("out_channels, kernel_size", [(10, 3), (5, 5)])
class TestConv1d:
    """
    This class implements the tests for the Conv1d.
    """

    @torch.no_grad()
    def test_forward(
        self,
        out_channels: int,
        kernel_size: int,
        inputs_1d: torch.Tensor,
    ) -> None:
        """
        This method is the test for the forward method.

        Args:
            out_channels: Output channels.
            kernel_size: Kernel size.
            inputs_1d: Inputs fixture. Dimensions: [batch size,
                number of input channels, sequence length].

        Returns:
            None.
        """

        # Rename fixture
        inputs: torch.Tensor = inputs_1d

        # Define module
        module: torch.nn.Module = Conv1d(inputs.shape[1], out_channels, kernel_size)
        module_test: torch.nn.Module = torch.nn.Conv1d(
            inputs.shape[1], out_channels, kernel_size, bias=False, dtype=torch.double
        )
        module_test.weight = module.weight

        # Compute outputs
        outputs: torch.Tensor = module(inputs)
        outputs_test: torch.Tensor = module_test(inputs)

        # Check shape
        assert (
            module.shape == module_test.weight.shape
        ), f"Incorrect shape, expected {module_test.weight.shape} and got {module.shape}"

        # Check values
        assert torch.allclose(outputs, outputs_test), "Incorrect forward values"

        return None

    @pytest.mark.parametrize("use_grad_outputs", [False, True])
    def test_backward(
        self,
        out_channels: int,
        kernel_size: int,
        use_grad_outputs: bool,
        inputs_1d: torch.Tensor,
    ) -> None:
        """
        This method is the test for the backward method.

        Args:
            kernel_size: Kernel size.
            num_groups: Number of groups.
            use_grad_outputs: Indicator to use grad outputs.
            inputs_1d: Inputs fixture. Dimensions: [batch size,
                number of channels, sequence length].

        Raises:
            RuntimeError: Error in gradient computation.

        Returns:
            None.
        """

        # Rename fixture
        inputs: torch.Tensor = inputs_1d
        inputs.requires_grad_(True)

        # Define module
        module: torch.nn.Module = Conv1d(inputs.shape[1], out_channels, kernel_size)
        module_test: torch.nn.Module = torch.nn.Conv1d(
            inputs.shape[1], out_channels, kernel_size, bias=False, dtype=torch.double
        )
        with torch.no_grad():
            module_test.weight = module.weight

        # Compute outputs and weights
        outputs: torch.Tensor = module(inputs)
        weights: torch.Tensor
        if use_grad_outputs:
            weights = torch.rand_like(outputs)
        else:
            weights = torch.ones_like(outputs)

        # Compute gradients
        (outputs * weights).sum().backward()
        if (inputs.grad is None) or (module.weight.grad is None):
            raise RuntimeError("Error in gradient computation")
        gradients_inputs: torch.Tensor = inputs.grad.clone()
        gradients_weight: torch.Tensor = module.weight.grad
        inputs.grad.zero_()

        # Compute torch gradients
        outputs = module_test(inputs)
        (outputs * weights).sum().backward()
        if (inputs.grad is None) or (module.weight.grad is None):
            raise RuntimeError("Error in gradient computation")
        gradients_inputs_test: torch.Tensor = inputs.grad.clone()
        gradients_weight_test: torch.Tensor = module.weight.grad
        inputs.grad.zero_()

        # Check gradients inputs values
        assert torch.allclose(
            gradients_inputs, gradients_inputs_test
        ), "Incorrect gradients value for inputs"

        # Check gradients weight values
        assert torch.allclose(
            gradients_weight, gradients_weight_test
        ), "Incorrect gradients value for weight"

        return None
