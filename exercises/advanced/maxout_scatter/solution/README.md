# Maxout scatter forward and backward (1 point)

Implement the forward and backward pass of a Maxout layer using
`torch.autograd.Function`.

The layer receives inputs with shape `[batch, input_dim]`, evaluates
`num_units` affine transformations for each output feature, and returns the
maximum value over those units. In the backward pass, use `scatter` to route
each output gradient only to the affine unit that won in the forward pass.

