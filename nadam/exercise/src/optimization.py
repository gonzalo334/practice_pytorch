# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict


class NAdam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the NAdam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 0.004,
    ) -> None:
        """
        This is the constructor for NAdam.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
            betas: betas for Adam. Defaults to (0.9, 0.999).
            eps: epsilon for approximation. Defaults to 1e-8.
            weight_decay: weight decay. Defaults to 0.0.
            momentum_decay: momentum decay. Defaults to 0.004.
        """

        # TODO

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
