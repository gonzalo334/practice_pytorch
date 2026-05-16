# Conv3D With Loops (2 points)

Implement the forward and backward pass of a `Conv3d` layer using
`torch.autograd.Function`.

The implementation must compute the convolution with explicit Python loops
over the batch, channels, output volume, and kernel volume. Do not use
`torch.nn.Conv3d`, `torch.nn.functional.conv3d`, unfold/fold helpers, or any
other built-in convolution operation inside `src.conv3d_loops`.

