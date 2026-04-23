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

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay
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
            lr: torch.Tensor = group["lr"]
            lr_decay: torch.Tensor = group["lr_decay"]
            weight_decay: torch.Tensor = group["weight_decay"]
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
                    param_state["t"] = torch.tensor(0.0)
                    param_state["state_sum"] = torch.zeros_like(parameter.data)

                # increment t
                param_state["t"] += 1

                # compute lr hat
                lr_hat: torch.Tensor = lr / (1 + (param_state["t"] - 1) * lr_decay)

                # compute state sum
                param_state["state_sum"] = (
                    param_state["state_sum"] + parameter_grad**2
                )

                # compute step parameter
                parameter.data -= lr_hat * (
                    parameter_grad / (torch.sqrt(param_state["state_sum"]) + eps)
                )

        return None
