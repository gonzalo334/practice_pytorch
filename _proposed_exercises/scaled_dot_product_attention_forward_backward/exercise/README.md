# Scaled dot-product attention forward and backward (2 points)

Implement the forward and backward pass of single-head scaled dot-product
attention using `torch.autograd.Function`.

The layer receives queries `query` with shape `[batch, target_length, dim]`,
keys `key` with shape `[batch, source_length, dim]`, and values `value` with
shape `[batch, source_length, value_dim]`. It must return an output tensor with
shape `[batch, target_length, value_dim]`.

The forward pass computes:

`scores = query @ key.transpose(-2, -1) / sqrt(dim)`

`weights = softmax(scores, dim=-1)`

`output = weights @ value`

When `causal=True`, positions must not attend to future source positions. The
tests use equal target and source lengths for the causal case.

The backward pass must return gradients with respect to `query`, `key`, and
`value`.
