# RNN forward and backward with embedding input (1.5 points)

Implement the forward and backward pass of a tanh RNN with embedding inputs
using `torch.autograd.Function`.

The input is a tensor of token ids with shape `[batch, sequence length]`. At
each time step the layer looks up the token embedding and computes:

`h_t = tanh(x_t W_ih^T + h_{t-1} W_hh^T + b_h)`

The `truncate_steps` argument simulates truncated backpropagation through time:
when it is greater than zero, gradients through the hidden state must stop at
time steps that are multiples of `truncate_steps`.

