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

        if lr < 0:
            raise ValueError("Invalid learning rate")
        if eps < 0:
            raise ValueError("Invalid epsilon value")
        if alpha < 0:
            raise ValueError("Invalid alpha value")
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value")
        if momentum < 0:
            raise ValueError("Invalid momentum value")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.
        """

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad.data
                if weight_decay != 0:
                    grad = grad.add(parameter.data, alpha=weight_decay)

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(parameter.data)
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter.data)
                    if centered:
                        state["grad_avg"] = torch.zeros_like(parameter.data)

                square_avg = state["square_avg"]
                state["step"] += 1
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt().add_(eps)
                else:
                    avg = square_avg.sqrt().add_(eps)

                if momentum > 0:
                    buffer = state["momentum_buffer"]
                    buffer.mul_(momentum).addcdiv_(grad, avg)
                    parameter.data.add_(buffer, alpha=-lr)
                else:
                    parameter.data.addcdiv_(grad, avg, value=-lr)

        return None

