# deep learning libraries
import torch

# other libraries
from typing import Iterator, Dict, Any, DefaultDict


class Adagrad(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

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
        lr: float = 1e-3,
        lr_decay: float = 0.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for Adagrad.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
            lr_decay: decay of learning rate. Defaults to 0.
            eps: epsilon value for avoiding overflow. Defaults to 0.
            weight_decay: weight decay rate. Defaults to 0.
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
