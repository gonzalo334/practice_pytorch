import torch


class AveragePool2d(torch.nn.Module):
    """
    Custom implementation of Average Pooling 2D.

    Args:
        kernel_size: size of the pooling window (int)
        stride: stride of the pooling window (int)
        padding: implicit zero padding (int)
    """

    def __init__(self, kernel_size: int, stride: int, padding: int = 0) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: tensor of shape [batch, channels, height, width]

        Returns:
            Outputs tensor. Dimensions: [batch, channels,
            (height + 2*padding - kernel_size) // stride + 1,
            (width + 2*padding - kernel_size) // stride + 1].
        """

        # TODO:
        B, Cin, H, W = inputs.shape
        K = self.kernel_size
        Hout = (H + 2*self.padding - K) // self.stride + 1
        Wout = (W + 2*self.padding - K) // self.stride + 1
        """Fold version"""
        # inputs_unfolded = torch.nn.functional.unfold(inputs, K, padding=self.padding, stride=self.stride)
        # B, Cin*K*K, Hout*Wout == inputs_unfolded.shape
        # inputs_windows = inputs_unfolded.view(B, Cin, K*K, Hout*Wout)
        # inputs_windows = torch.mean(inputs_windows, dim=2)
        # outputs = inputs_windows.view(B, Cin, Hout, Wout)

        """Loops version"""
        padded_inputs = torch.zeros((B,Cin,H + 2 * self.padding,W + 2 * self.padding,),dtype=inputs.dtype)
        padded_inputs[:,:,self.padding:self.padding + H,self.padding:self.padding + W,] = inputs
        outputs = torch.zeros((B, Cin, Hout, Wout), dtype=inputs.dtype)
        for hout in range(Hout):
            for wout in range(Wout):
                h_start = hout * self.stride
                w_start = wout * self.stride
                window = padded_inputs[:, :, h_start:h_start+K, w_start:w_start+K]
                outputs[:, :, hout, wout] = torch.mean(window, dim=(2,3))
        return outputs



        