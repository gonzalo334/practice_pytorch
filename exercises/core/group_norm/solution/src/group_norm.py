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

        # compute number of channels per group and original shape
        original_shape: torch.Size = inputs.shape

        # reshape inputs
        inputs = inputs.view(original_shape[0], self.num_groups, -1)

        # compute mean and std
        means = torch.mean(inputs, dim=2, keepdim=True)
        vars_ = torch.var(inputs, dim=2, keepdim=True)

        # compute norm
        outputs: torch.Tensor = (inputs - means) / torch.sqrt(vars_ + self.eps)

        # compute affine transformation
        if self.affine:
            outputs = outputs.view(
                original_shape[0], original_shape[1], -1
            ) * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(
                0
            ).unsqueeze(
                2
            )

        # reshape to original dim
        outputs = outputs.view(original_shape)

        return outputs

    def reset_parameters(self) -> None:
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
