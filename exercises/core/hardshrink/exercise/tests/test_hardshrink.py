"""
This module contains the code for testing the Hardshrink.
"""

# 3pps
import torch
import pytest

# Own libraries
from src.hardshrink import Hardshrink


class TestHardShrinnk:
    """
    This class contains the tests for Hardshrink layer.
    """
    
    @pytest.mark.order(2)
    @pytest.mark.parametrize("lambd", [0.5, 1.0])
    def test_hardshrink_forward(self, lambd: float, inputs: torch.Tensor) -> None:
        """
        This function is to test the Hardshrink forward pass.
        
        Args:
            lambd: parameter for Hardshrink.
            inputs: inputs tensor from fixture.

        Returns:
            None.
        """

        # Define modules
        module: torch.nn.Module = Hardshrink(lambd)
        module_torch: torch.nn.Module = torch.nn.Hardshrink(lambd)

        # Compute outputs
        outputs: torch.Tensor = module(inputs)
        outputs_torch: torch.Tensor = module_torch(inputs)

        # Check shape
        assert outputs.shape == outputs_torch.shape, (
            f"Incorrect output shape, expected {outputs_torch.shape} and got "
            f"{outputs.shape}"
        )

        # Check outputs value
        assert torch.allclose(outputs, outputs_torch), "Incorrect outputs value"
        
        # iter over limit values
        value: float
        for value in (-lambd, lambd):
            # Change values
            with torch.no_grad():
                inputs[:] = value

            # Compute outputs
            outputs = module(inputs)
            outputs_torch = module_torch(inputs)

            # Check outputs value
            assert torch.allclose(outputs, outputs_torch), (
                f"Incorrect outputs value at {value}"
            )

        return None

    @pytest.mark.order(3)
    @pytest.mark.parametrize("lambd", [0.5, 1.0])
    def test_hardshrink_backward(self, lambd: float, inputs: torch.Tensor) -> None:
        """
        This function is to test the Hardshrink backward pass.
        
        Args:
            lambd: parameter for Hardshrink.
            inputs: inputs tensor from fixture.

        Returns:
            None.
        """

        # Activate gradients
        inputs.requires_grad_(True)

        # Define modules
        module: torch.nn.Module = Hardshrink(lambd)
        module_torch: torch.nn.Module = torch.nn.Hardshrink(lambd)

        # Compute outputs backward
        outputs: torch.Tensor = module(inputs)
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()
        inputs.grad.zero_()

        # Compute torch outputs backward
        outputs_torch: torch.Tensor = module_torch(inputs)
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch = inputs.grad.clone()
        inputs.grad.zero_()

        # Check shape
        assert grad_inputs.shape == grad_inputs_torch.shape, (
            f"Incorrect gradients inputs shape, expected {grad_inputs_torch.shape} "
            f"and got {grad_inputs.shape}"
        )

        # Check outputs value
        assert torch.allclose(
            grad_inputs, grad_inputs_torch
        ), "Incorrect gradients inputs value"
        
        # iter over limit values
        value: float
        for value in (-lambd, lambd):
            # Change values
            with torch.no_grad():
                inputs[:] = value

            # Compute outputs backward
            outputs = module(inputs)
            outputs.sum().backward()
            if inputs.grad is None:
                assert False, "Gradients not returned, none value detected"
            grad_inputs = inputs.grad.clone()
            inputs.grad.zero_()

            # Compute torch outputs backward
            outputs_torch = module_torch(inputs)
            outputs_torch.sum().backward()
            if inputs.grad is None:
                assert False, "Gradients not returned, none value detected"
            grad_inputs_torch = inputs.grad.clone()
            inputs.grad.zero_()

            # Check outputs value
            assert torch.allclose(
                grad_inputs, grad_inputs_torch
            ), f"Incorrect gradients inputs value at {value}"

        return None
