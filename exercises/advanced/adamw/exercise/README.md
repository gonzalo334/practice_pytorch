# AdamW optimizer (1 point)

Implement the AdamW optimizer by subclassing `torch.optim.Optimizer`.

The optimizer must match `torch.optim.AdamW` for the tested learning rate,
betas, epsilon, weight decay, and AMSGrad settings.

Each parameter tensor can have any shape. The optimizer state must keep the
step count, first moment estimate, second moment estimate, and when
`amsgrad=True`, the maximum second moment estimate.
