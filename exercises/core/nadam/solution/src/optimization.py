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

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # iter over groups
        for group in self.param_groups:
            # get parameters
            b1: torch.Tensor
            b2: torch.Tensor
            b1, b2 = group["betas"]
            lr: torch.Tensor = group["lr"]
            weight_decay: torch.Tensor = group["weight_decay"]
            momentum_decay: torch.Tensor = group["momentum_decay"]
            eps = group["eps"]

            # iter over parameters
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                # compute parameter grad
                parameter_grad: torch.Tensor = parameter.grad.data + (
                    weight_decay * parameter.data
                )

                # get parameter state
                param_state: Dict[str, torch.Tensor] = self.state[parameter]

                # init parameters
                if "t" not in param_state:
                    param_state["ut_acc"] = torch.ones_like(parameter.data)
                    param_state["ut2_acc"] = torch.ones_like(parameter.data)
                    param_state["mt"] = torch.zeros_like(parameter.data)
                    param_state["vt"] = torch.zeros_like(parameter.data)
                    param_state["t"] = torch.tensor(0.0)

                # compute mt and vt
                param_state["mt"] = b1 * param_state["mt"] + (1 - b1) * parameter_grad
                param_state["vt"] = b2 * param_state["vt"] + (1 - b2) * (
                    parameter_grad**2
                )

                # update t
                param_state["t"] = param_state["t"] + 1

                # compute uts
                ut: torch.Tensor = b1 * (
                    1 - 0.5 * 0.96 ** (param_state["t"] * momentum_decay)
                )
                param_state["ut_acc"] = param_state["ut_acc"] * ut
                ut2: torch.Tensor = b1 * (
                    1 - 0.5 * 0.96 ** ((param_state["t"] + 1) * momentum_decay)
                )
                param_state["ut2_acc"] = param_state["ut2_acc"] * ut2

                # compute hat variables
                mt_hat: torch.Tensor = (ut2 * param_state["mt"]) / (
                    1 - param_state["ut2_acc"]
                ) + (1 - ut) * parameter_grad / (1 - param_state["ut_acc"])
                vt_hat: torch.Tensor = param_state["vt"] / (1 - b2 ** param_state["t"])

                # compute step parameter
                parameter.data = parameter.data - lr * (
                    mt_hat / (torch.sqrt(vt_hat) + eps)
                )

        return None
