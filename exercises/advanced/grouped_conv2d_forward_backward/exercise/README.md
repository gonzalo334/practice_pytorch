# Grouped Conv2d forward and backward (1 point)

Implement the forward and backward pass of a grouped 2D convolution using
`torch.autograd.Function`.

The custom layer must match `torch.nn.Conv2d` when `groups > 1`, including
gradients with respect to inputs, weights and bias.

