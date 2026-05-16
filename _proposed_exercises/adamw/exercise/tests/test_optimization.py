import copy

import torch
import pytest

from src.optimization import AdamW


@pytest.mark.parametrize(
    "lr, betas, eps, weight_decay, amsgrad",
    [
        (1e-3, (0.9, 0.999), 1e-8, 1e-2, False),
        (5e-4, (0.8, 0.95), 1e-6, 1e-1, True),
    ],
)
def test_adamw(
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    amsgrad: bool,
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

    optimizer_torch = torch.optim.AdamW(
        model_torch.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=False,
    )
    optimizer_custom = AdamW(
        model_custom.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
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
