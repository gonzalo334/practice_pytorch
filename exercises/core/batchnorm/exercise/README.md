# BatchNorm2d (3 points)

Here you will have to code the forward and backward of custom version of the
BatchNorm2d. You will only code the eval version, which means that you will
use the running mean and var, instead of taking them from each batch. The
formula of the custom version is the following:

```math
y = \frac{LeakyReLU(x - E(x))}{\sigma^2(x) + \epsilon}
```

where $E(x)$ is the running mean and $\sigma^2(x)$ is the running variance.

As you can check, there is also a second addition, you will have to introduce
a LeakyReLU after subtracting the mean. Remember that you cannot use any
function from the nn package, so you will have to code the LeakyReLU yourself
using masks.

To make it simpler, in this version we have ignored the affine transformations
(gamma and beta parameters) so you must ignore them.

### `forward` (1 points)

Here you will have to code the forward method.

### `backward` (2 points)

Here you will have to code the backward method.
