# deep learning libraries
import torch

# other libraries
from typing import Any, DefaultDict, Iterable


class AdamW(torch.optim.Optimizer):
    """
    This class is a custom implementation of AdamW.
    """

    param_groups: list[dict[str, Any]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        """
        This is the constructor for AdamW.
        """

        # TODO
        raise NotImplementedError

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        Perform one AdamW optimization step.
        """

        # TODO
        raise NotImplementedError
