## Grouped MaxPool2d (6 points)

Here you will have to implement a custom MaxPool2d without using loops. First, let's explain the custom MaxPool2d operation. Usually, MaxPool2d is computed without any depth, that means that the output always has the same number of channels that the input. However, here we want to merge the concept of groups with the MaxPool2d. Therefore, now the max operation will be computed from all the channels in the same group.

- Normal MaxPool2d:
```
Inputs: [batch size, channels, height, width]
Outputs: [batch size, channels, height - kernel size + 1, width - kernel size + 1].
```

- Grouped MaxPool2d with 1 group:
```
Inputs: [batch size, channels, height, width]
Outputs: [batch size, 1, height - kernel size + 1, width - kernel size + 1].
```
- Grouped MaxPool2d with n groups:
```
Inputs: [batch size, channels, height, width]
Outputs: [batch size, n, height - kernel size + 1, width - kernel size + 1].
```

You will have to implement this without using any loops or any function from the nn package (torch.where is not allowed either), besides unfold, fold and one_hot. You may want look at the following functions:

    fold, unfold, permute, one_hot, max

You should not worry about stride, dilation or padding.

### `forward` (2 points)

Here you will have to code the forward method.

### `backward` (4 points)

Here you will have to code the backward method.
