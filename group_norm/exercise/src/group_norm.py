# deep learning libraries
import torch


class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        This is the constructor of the GroupNorm class.

        Args:
            num_groups: number of groups to use.
            num_channels: number of channels to use.
            eps: epsilon to avoid overflow. Defaults to 1e-5.
            affine: Indicator to perform affine transformation.
                Defaults to True.
        """

        # call super class constructor
        super().__init__()

        # save attributes
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        # create parameters if it is an affine transformation
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, dtype=dtype))

        self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass of the module.

        Args:
            inputs: input tensor. Dimensions: [batch, channels, *].

        Returns:
            outputs tensor. Dimensions: [batch, channels, *].
        """
        G = self.num_groups
        B, Cin = inputs.shape[0:2]
        outputs = torch.zeros_like(inputs, dtype=inputs.dtype)
        inputs_grouped = inputs.view(B, G, Cin // G, -1)
        var = torch.var(inputs_grouped, dim=(2,3)).view(B, G, 1, 1)
        mean = torch.mean(inputs_grouped, dim=(2,3)).view(B, G, 1, 1)
        denominator = torch.sqrt(var + self.eps)
        outputs = (inputs_grouped - mean) / denominator
        if self.affine:
            outputs = outputs.view(B, Cin, -1) * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)
        return outputs.view(inputs.shape)
    
    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
