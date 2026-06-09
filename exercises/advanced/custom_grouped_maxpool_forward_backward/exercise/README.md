# Custom grouped maxpool forward and backward (1 point)

Implement a grouped max-pooling layer using `torch.autograd.Function`.

The layer receives an input with shape `[batch, channels, height, width]`,
splits the channels into `num_groups`, and for each group computes the maximum
over all channels in the group and over each spatial pooling window. The output
shape is `[batch, num_groups, out_height, out_width]`.

