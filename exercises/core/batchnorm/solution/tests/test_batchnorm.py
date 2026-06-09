"""
This module contains the code to test the BatchNorm2d.
"""

# 3pps
import torch
import pytest

# Own modules
from src.batchnorm import BatchNorm2d


@pytest.mark.parametrize("negative_slope", [0.01, 0.5])
class TestBatchNorm2d:
    @torch.no_grad()
    def test_forward(self, negative_slope: float, inputs_2d: torch.Tensor) -> None:
        """
        This method tests the forward method.

        Args:
            inputs_2d: Inputs with 2d shape. Dimensions: [batch,
                channels, height, width].

        Returns:
            None.
        """

        # Rename fixture
        inputs: torch.Tensor = inputs_2d

        # Define torch module
        module_torch: torch.nn.Module = torch.nn.BatchNorm2d(
            inputs.shape[1], affine=False, dtype=torch.double
        )
        module_torch.train()
        _ = module_torch(inputs)
        module_torch.eval()

        # Define module
        module: torch.nn.Module = BatchNorm2d(inputs.shape[1], negative_slope)
        module.running_mean = module_torch.running_mean.clone()
        module.running_var = module_torch.running_var

        # Redefine torch
        module_torch = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope), module_torch
        )
        module_torch[1].running_mean[:] = 0

        # Iter over cases
        for i in range(2):
            # Set elements to delta in second iteration
            if i == 1:
                inputs[:] = 0

            # Compute outputs
            outputs: torch.Tensor = module(inputs)
            outputs_torch: torch.Tensor = module_torch(
                inputs - module.running_mean.view(1, -1, 1, 1)
            )

            # Check values
            if i == 0:
                assert torch.allclose(
                    outputs, outputs_torch
                ), "Incorrect forward values"
            else:
                assert torch.allclose(
                    outputs, outputs_torch
                ), "Incorrect forward values at zero"

        return None

    def test_backward(self, negative_slope: float, inputs_2d: torch.Tensor) -> None:
        """
        _summary_

        Args:
            negative_slope: _description_
            inputs_2d: _description_

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            _description_
        """

        # Rename fixture
        inputs: torch.Tensor = inputs_2d

        # Define torch module
        module_torch: torch.nn.Module = torch.nn.BatchNorm2d(
            inputs.shape[1], affine=False, dtype=torch.double
        )
        module_torch.train()
        _ = module_torch(inputs)
        module_torch.eval()

        # Define module
        module: torch.nn.Module = BatchNorm2d(inputs.shape[1], negative_slope)
        module.running_mean = module_torch.running_mean.clone()
        module.running_var = module_torch.running_var

        # Redefine torch
        module_torch = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope), module_torch
        )
        module_torch[1].running_mean[:] = 0

        # Activate gradients
        inputs.requires_grad_(True)

        # Iter over cases
        for i in range(2):
            # Set elements to delta in second iteration
            if i == 1:
                with torch.no_grad():
                    inputs[:] = 0

            # Compute outputs
            outputs: torch.Tensor = module(inputs)
            outputs.sum().backward()
            if inputs.grad is None:
                raise RuntimeError("Error in gradient computation")
            grad_inputs: torch.Tensor = inputs.grad.clone()
            inputs.grad.zero_()

            # Compute torch gradients
            outputs = module_torch(inputs - module.running_mean.view(1, -1, 1, 1))
            outputs.sum().backward()
            if inputs.grad is None:
                raise RuntimeError("Error in gradient computation")
            grad_inputs_torch: torch.Tensor = inputs.grad.clone()
            inputs.grad.zero_()

            # Check grad inputs value
            if i == 0:
                assert torch.allclose(
                    grad_inputs, grad_inputs_torch
                ), "Incorrect backward values"
            else:
                assert torch.allclose(
                    grad_inputs, grad_inputs_torch
                ), "Incorrect backward values at zero"

        return None
