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

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        Perform one AdamW optimization step.
        """

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                param.data = param.data * (1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt() / bias_correction2**0.5 + eps
                else:
                    denom = exp_avg_sq.sqrt() / bias_correction2**0.5 + eps

                param.data = param.data - lr * exp_avg / bias_correction1 / denom
