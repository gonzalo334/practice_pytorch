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
        defaults = {"lr":lr, "eps":eps, "betas":betas, "weight_decay":weight_decay, "momentum_decay":momentum_decay} 
        super().__init__(params, defaults)
    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum_decay = group["momentum_decay"]
            for param in group["params"]:

                grad = param.grad + weight_decay*param.data
                param_state = self.state[param]
                if "step" not in param_state:
                    param_state["step"] = 0
                    param_state["mt"] = torch.zeros_like(param.data)
                    param_state["vt"] = torch.zeros_like(param.data)
                    param_state["acc_mu"] = 1.0

                param_state["step"] += 1
                
                mu_t = b1 * (1 - 1/2 * (0.96)**(param_state["step"]*momentum_decay))
                mu_tnext = b1 * (1 - 1/2 * (0.96)**((param_state["step"]+1)*momentum_decay))                

                param_state["acc_mu"] *= mu_t

                param_state["mt"] = b1*param_state["mt"] + (1-b1)*grad
                param_state["vt"] = b2*param_state["vt"] + (1-b2)*(grad)**2

                mt_norm = mu_tnext*param_state["mt"] / (1 - (param_state["acc_mu"]*mu_tnext)) + (1-mu_t)*grad/(1 - param_state["acc_mu"])
                vt_norm = param_state["vt"] / (1 - b2**param_state["step"])
                param.data = param.data - lr*mt_norm/(torch.sqrt(vt_norm) + eps)
