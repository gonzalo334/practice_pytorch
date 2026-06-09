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


class TestCustomMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size: int, num_groups: int) -> None:
        # Call super class constructor
        super().__init__()

        # Set attributes
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Compute number of channels per groups
        num_channels_group: int = inputs.shape[1] // self.num_groups

        # Set outputs sizes
        output_size: int = inputs.shape[2] - self.kernel_size + 1

        # Init outputs with nans
        outputs = torch.zeros(
            (inputs.shape[0], self.num_groups, output_size, output_size),
            dtype=torch.double,
        )

        # Iter over spatial dimensions
        for i in range(output_size):
            for j in range(output_size):
                # Iter over groups
                for k in range(self.num_groups):
                    # Compute conv operation
                    outputs[:, k, i, j] = torch.amax(
                        inputs[
                            :,
                            k * num_channels_group : k * num_channels_group
                            + num_channels_group,
                            i : i + self.kernel_size,
                            j : j + self.kernel_size,
                        ],
                        dim=(1, 2, 3),
                    )

        return outputs
