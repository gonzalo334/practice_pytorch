import copy

import torch
import pytest

from src.optimization import RMSprop


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "lr, alpha, eps, weight_decay, momentum, centered",
    [
        (1e-2, 0.99, 1e-8, 0.0, 0.0, False),
        (1e-3, 0.9, 1e-6, 1e-2, 0.5, True),
    ],
)
def test_rmsprop(
    lr: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
) -> None:
    torch.manual_seed(0)
    original = torch.nn.Sequential(
        torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )
    model_torch = copy.deepcopy(original)
    model_custom = copy.deepcopy(original)

    inputs = torch.randn(8, 5)
    targets = torch.randn(8, 2)
    loss_fn = torch.nn.MSELoss()

    optimizer_torch = torch.optim.RMSprop(
        model_torch.parameters(),
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
    )
    optimizer_custom = RMSprop(
        model_custom.parameters(),
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
    )

    for _ in range(5):
        optimizer_torch.zero_grad()
        loss_torch = loss_fn(model_torch(inputs), targets)
        loss_torch.backward()
        optimizer_torch.step()

        optimizer_custom.zero_grad()
        loss_custom = loss_fn(model_custom(inputs), targets)
        loss_custom.backward()
        optimizer_custom.step()

    for param_torch, param_custom in zip(
        model_torch.parameters(), model_custom.parameters()
    ):
        assert torch.allclose(param_torch, param_custom, atol=1e-6)

