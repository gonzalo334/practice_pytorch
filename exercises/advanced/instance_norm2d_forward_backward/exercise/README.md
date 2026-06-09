# InstanceNorm2d forward and backward (1 point)

Implement the forward and backward pass of `InstanceNorm2d` using
`torch.autograd.Function`.

The input tensor has shape `[batch, channels, height, width]`. The learnable
scale `weight` and offset `bias` have shape `[channels]`. For each sample and
channel, normalize over the spatial dimensions `[height, width]`.

The output must have the same shape as the input and must match
`torch.nn.InstanceNorm2d` with `affine=True` and `track_running_stats=False`,
including gradients with respect to inputs, weight, and bias.
