# AdaptiveAvgPool2d forward and backward (1 point)

Implement the forward and backward pass of `AdaptiveAvgPool2d` using
`torch.autograd.Function`.

The input tensor has shape `[batch, channels, input_height, input_width]`.
The layer receives an `output_size` tuple `(output_height, output_width)` and
returns a tensor with shape `[batch, channels, output_height, output_width]`.

Each output location averages the corresponding adaptive input bin. The custom
layer must match `torch.nn.AdaptiveAvgPool2d`, including gradients with respect
to the input.
