# deep learning libraries
import torch

# other libraries
import copy
import pytest

# own modules
from src.optimization import NAdam
from src.utils import set_seed


@pytest.mark.order(7)
@pytest.mark.parametrize(
    "lr, betas, weight_decay, momentum_decay",
    [(1e-3, (0.9, 0.999), 0, 0), (1e-4, (0.5, 0.4), 1e-2, 1e-1)],
)
def test_nadam(
    lr: float, betas: tuple[float, float], weight_decay: float, momentum_decay: float
) -> None:
    """
    This function is the test for the NAdam algorithm.

    Args:
        lr: learning rate.
        betas: betas parameters.
        weight_decay: weight decay rate.
        momentum_decay: momentum decay.
    """

    for seed in range(10):
        # set seed
        set_seed(42)

        # define model
        model_original: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(30, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )

        # clone model
        model1: torch.nn.Module = copy.deepcopy(model_original)
        model2: torch.nn.Module = copy.deepcopy(model_original)

        # define inputs and targets
        inputs: torch.Tensor = torch.rand(64, 30)
        targets: torch.Tensor = torch.rand(64, 1)

        # define loss and lr
        loss = torch.nn.L1Loss()

        # define optimizers
        optimizer1: torch.optim.Optimizer = torch.optim.NAdam(
            model1.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
        )
        optimizer2: torch.optim.Optimizer = NAdam(
            model2.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
        )

        # optimize first model
        for _ in range(10):
            outputs: torch.Tensor = model1(inputs)
            loss_value: torch.Tensor = loss(outputs, targets)
            optimizer1.zero_grad()
            loss_value.backward()
            optimizer1.step()

        # optimize second model
        for _ in range(10):
            # optimize second model
            outputs = model2(inputs)
            loss_value = loss(outputs, targets)
            optimizer2.zero_grad()
            loss_value.backward()
            optimizer2.step()

        # check parameters of both models
        for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(
                parameter1.data, parameter2.data, atol=1e-3
            ), "Incorrect return of the algorithm"

    return None
