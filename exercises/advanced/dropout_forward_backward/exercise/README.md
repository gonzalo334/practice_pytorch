# Dropout forward and backward (0.5 points)

Implement the forward and backward pass of inverted dropout using
`torch.autograd.Function`.

During training, elements are randomly zeroed with probability `p` and the
remaining values are scaled by `1 / (1 - p)`. During evaluation, the layer must
return the input unchanged.

