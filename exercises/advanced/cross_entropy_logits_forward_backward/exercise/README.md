# Cross entropy from logits forward and backward (1 point)

Implement the forward and backward pass of cross entropy loss from logits using
`torch.autograd.Function`.

The logits tensor has shape `[batch, num_classes]`. The target tensor has shape
`[batch]` and contains class indices. The implementation must be numerically
stable and support `ignore_index` plus `reduction="none"`, `"sum"`, and
`"mean"`.

For `reduction="none"`, return a tensor with shape `[batch]`. For `"sum"` and
`"mean"`, return a scalar tensor. The custom loss must match
`torch.nn.functional.cross_entropy`, including gradients with respect to the
logits.
