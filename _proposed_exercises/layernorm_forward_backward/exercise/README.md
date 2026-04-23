# LayerNorm forward and backward (1 point)

Implement the forward and backward pass of Layer Normalization using
`torch.autograd.Function`.

The custom layer must normalize over the last `normalized_shape` dimensions and
must match `torch.nn.LayerNorm`, including gradients with respect to inputs,
weight and bias.

