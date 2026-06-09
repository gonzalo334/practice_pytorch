"""
This module contains auxiliary code for the tests.
"""

# Standard libraries
import os
import random
from typing import Any

# 3pps
import torch


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix randomness.
    """

    # set seed in random
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def add_seed(parameters: tuple[Any, ...], num_seeds: int = 3) -> list[tuple[Any, ...]]:
    """
    This function

    Args:
        parameters: _description_
        num_seeds: _description_. Defaults to 3.

    Returns:
        _description_
    """

    return [(*parameters, seed) for seed in range(num_seeds)]


class MaxOutLoop(torch.nn.Module):
    """
    This an implementation of the MaxOut unit with a for loop for a
    generic number of units.

    Attributes:
        num_units: Number of units.
        linears: list of linear layers.
    """

    def __init__(self, num_units: int, input_dim: int, output_dim: int) -> None:
        # Call super class
        super().__init__()

        # Set attributes
        self.num_units = num_units

        # Init layer
        self.linears: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, output_dim, bias=False, dtype=torch.double)
            for _ in range(self.num_units)
        ]

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Repeat inputs
        outputs_list: list[torch.Tensor] = [
            self.linears[i](inputs) for i in range(self.num_units)
        ]
        outputs: torch.Tensor = torch.stack(outputs_list)

        # Compute max accross units
        outputs = torch.amax(outputs, dim=0)

        return outputs

    @torch.no_grad()
    def load_parameters(self, maxout: torch.nn.Module) -> None:
        """
        This function loads the parameters from a MaxOut without for
        loops.

        Args:
            maxout: MaxOut layer without loops. It has only one linear
                layer with a weight of dimensions: [number of units,
                output features, input features].

        Returns:
            None.
        """

        for i, linear in enumerate(self.linears):
            linear.weight.data = maxout.weight[i].data
