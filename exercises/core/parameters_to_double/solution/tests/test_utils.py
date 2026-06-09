# deep learning libraries
import torch

# own modules
from src.utils import parameters_to_double


def test_parameters_to_double() -> None:
    """
    This function is the test for the parameters_to_double utility.
    """

    # define model
    model: torch.nn.Module = torch.nn.Sequential(
        torch.nn.Linear(5, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1)
    )

    # convert parameters
    parameters_to_double(model)

    # check parameters
    for parameter in model.parameters():
        assert parameter.dtype == torch.float64, "Incorrect parameter dtype"

    return None
