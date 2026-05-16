# EmbeddingBag forward and backward (1 point)

Implement the forward and backward pass of an `EmbeddingBag` layer using
`torch.autograd.Function`.

The layer receives a one-dimensional `inputs` tensor of token ids with shape
`[num_indices]` and a one-dimensional `offsets` tensor with shape `[num_bags]`.
The embedding table `weight` has shape `[num_embeddings, embedding_dim]`.

For each bag `i`, the token ids are taken from
`inputs[offsets[i]:offsets[i + 1]]`; the final bag ends at `num_indices`. The
output has shape `[num_bags, embedding_dim]`.

Support `mode="sum"` and `mode="mean"`. Repeated token ids must accumulate
gradients correctly, and the custom layer must match `torch.nn.EmbeddingBag`.
