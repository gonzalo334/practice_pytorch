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
        defaults = {"lr": lr, "alpha": alpha, "eps": eps, "weight_decay": weight_decay, "momentum":momentum, "centered":centered}
        super().__init__(params, defaults)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.
        """

        # TODO
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]
            for param in group["params"]:
                
                param_state = self.state[param]

                if "vt" not in param_state:
                    param_state["vt"] = torch.zeros_like(param.data)
                    param_state["bt"] = torch.zeros_like(param.data)
                    param_state["gt_ave"] = torch.zeros_like(param.data)

                grad = param.grad + weight_decay * param.data

                vt = alpha * param_state["vt"] + (1 - alpha) * (grad**2)
                param_state["vt"] = vt
                vt_n = vt 

                if centered:
                    gt_ave = param_state["gt_ave"] * alpha + (1 - alpha) * grad
                    vt_n = vt_n - (gt_ave**2)

                    param_state["gt_ave"] = gt_ave

                bt =  momentum * param_state["bt"] + grad / (torch.sqrt(vt_n) + eps)
                param_state["bt"] = bt
                if momentum > 0:
                    param.data = param.data - lr * bt
                else:
                    param.data = param.data - lr * grad / (torch.sqrt(vt_n) + eps)
