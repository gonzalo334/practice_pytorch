# GroupNorm Without Affine (6 points)

Implement the forward pass of `GroupNorm` in the `src.group_norm` module. This is the basic version of GroupNorm, so affine scaling and bias parameters are not required. You cannot use loops, the `nn` package, or `torch.where`; use indexing and tensor operations instead.
