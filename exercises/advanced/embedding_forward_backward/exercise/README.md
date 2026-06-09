# Embedding forward and backward (0.5 points)

Implement the forward and backward pass of an embedding layer using
`torch.autograd.Function`.

The custom layer must support repeated token ids and `padding_idx`, and must
match `torch.nn.Embedding` gradients with respect to the embedding weights.

