# deep learning libraries
import torch

# other libraries
from typing import Any, DefaultDict, Iterable


class RMSprop(torch.optim.Optimizer):
    """
    This class is a custom implementation of RMSprop.
    """

    param_groups: list[dict[str, Any]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        """
        This is the constructor for RMSprop.
        """

        # TODO

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.
        """

        # TODO

