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
        self.num_units = num_units

        self.weight = torch.nn.Parameter(
            torch.randn((num_units, output_dim, input_dim), dtype=torch.float64)
        )


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
        B, Din = inputs.shape
        K = self.num_units
        inputs_repeated = inputs.unsqueeze(1).repeat(1, K, 1)
        inputs_reshaped = inputs_repeated.view(K * B, 1, Din)
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
        B, Din = inputs.shape
        K, Dout, Din = self.weight.shape
        weight_view = self.weight.view(1, K, Dout, Din)
        weight_reshaped = weight_view.repeat(B, 1, 1, 1).view(B*K, Dout, Din)
        return weight_reshaped.transpose(1,2)



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
        inputs_reshaped = self.reshape_inputs(inputs) # B*K, 1, Din
        weights_reshaped = self.reshape_weight(inputs) # B*K, Din, Dout
        z = torch.bmm(inputs_reshaped, weights_reshaped) # B*K, 1, Dout
        h = z.view(B, self.num_units, -1) # B, K, Dout
        outputs = torch.amax(h, dim=1) # B, Dout
        return outputs

