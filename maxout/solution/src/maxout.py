"""
This module contains the code for MaxOut.
"""

# 3pps
import torch


class MaxOut(torch.nn.Module):
    """
    This class implements the MaxOut layer without loops.

    Attr:
        num_units: Number of linear layers the MaxOut s going to use.
        weight: Tensor object with all the weights of the different
            layers. Dimensions: [num_units, output dim, input dim].
            The dtype is a double.
    """

    # Define attributes
    num_units: int
    weight: torch.Tensor

    def __init__(self, num_units: int, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the class.
        
        Returns:
            None.
        """

        # Call super class
        super().__init__()

        # Set attributes
        self.num_units = num_units
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand((self.num_units, output_dim, input_dim), dtype=torch.double)
        )

        return None

    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method reshapes the inputs in a way later can be used to perform
        matrix multiplication.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Inputs reshaped. Dimensions: [number of units * batch size,
                1, input dim].
        """

        # Repeat inputs
        inputs = inputs.view(inputs.shape[0], 1, 1, inputs.shape[1]).clone()
        inputs = inputs.repeat(1, self.num_units, 1, 1)
        inputs = inputs.view(-1, *inputs.shape[2:])

        return inputs

    def reshape_weight(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function transform the dimensions of the weight so it can
        be multiplied with bmm.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Weights reshaped.
        """

        # Get weight
        weight: torch.Tensor = self.weight.view(1, *self.weight.shape).clone()
        weight = weight.repeat(inputs.shape[0], 1, 1, 1)
        weight = weight.view(-1, *weight.shape[2:])
        weight = weight.transpose(1, 2)

        return weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Output tensor. Dimensions: [batch size, output dim].
        """

        # Reshape inputs
        inputs_reshaped: torch.Tensor = self.reshape_inputs(inputs)

        # Get weight
        weight: torch.Tensor = self.reshape_weight(inputs)

        # Compute outputs
        outputs = torch.bmm(inputs_reshaped, weight)

        # Compute final shape
        outputs = outputs.view(inputs.shape[0], self.num_units, -1)

        # Compute max across units
        outputs = torch.amax(outputs, dim=1)

        return outputs
