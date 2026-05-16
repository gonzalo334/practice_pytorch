# GRUCell forward and backward (1.5 points)

Implement the forward and backward pass of a single-step GRU cell using
`torch.autograd.Function`.

The layer receives `inputs` with shape `[batch, input_dim]` and `hidden` with
shape `[batch, hidden_dim]`. The learnable parameters must match
`torch.nn.GRUCell`:

- `weight_ih` has shape `[3 * hidden_dim, input_dim]`.
- `weight_hh` has shape `[3 * hidden_dim, hidden_dim]`.
- `bias_ih` has shape `[3 * hidden_dim]`.
- `bias_hh` has shape `[3 * hidden_dim]`.

The output has shape `[batch, hidden_dim]` and must match `torch.nn.GRUCell`,
including gradients with respect to inputs, previous hidden state, weights, and
biases.
