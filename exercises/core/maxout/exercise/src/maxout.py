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

        # TODO
        self.weight = torch.nn.Parameter(
            torch.rand((num_units, output_dim, input_dim), dtype=torch.float64)
        )
        self.num_units = num_units

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

        # TODO
        K = self.num_units
        B, Din = inputs.shape
        inputs_reshaped = inputs.view(1,B,1,Din).repeat(K, 1, 1, 1).view(K*B, 1, Din)
        return inputs_reshaped

    def reshape_weight(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function transform the dimensions of the weight so it can
        be multiplied with bmm.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Weights reshaped.
        """

        # TODO
        K = self.num_units
        B, Din = inputs.shape
        K, Dout, Din = self.weight.shape
        weight_reshaped = self.weight.view(K, 1, Dout, Din).repeat(1, B, 1, 1).view(K*B, Dout, Din)
        return weight_reshaped.transpose(1,2) # K*B, Din, Dout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Output tensor. Dimensions: [batch size, output dim].
        """

        # TODO
        B, Din = inputs.shape
        K, Dout, Din = self.weight.shape
        inputs_reshaped = self.reshape_inputs(inputs) # K*B, 1, Din
        weight_reshaped = self.reshape_weight(inputs) #K*B, Din, Dout
        outputs_units = torch.bmm(inputs_reshaped, weight_reshaped).view(K, B, Dout) # K, B, Dout
        outputs = torch.amax(outputs_units, dim=(0)) 
        return outputs
