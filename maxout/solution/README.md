# MaxOut (7 points)

You have to code MaxOut layer in pytorch for a generic number of units. For
this implementation you cannot use loops, so everything has to be vectorized
operations. You have the formula and diagram of the MaxOut in the `artifacts`
folder. Therefore, we will use the
[batch matrix multiplication](https://pytorch.org/docs/stable/generated/torch.bmm.html)
(BMM). We advice to have a look at it, since understanding the formula and
shapes is a key step to complete the exercise. In this case you have to
complete the following functions:

### `__init__` (0.5 points)

Here you will have to define the constructor and the objects that will be used
in next methods.

### `reshape_inputs` (2 point)

This method reshapes the inputs in a way they can be used in the BMM. For this
function the target shape is provided in the docstring of the function. A test
is available to check the functionality of the method.

Hint: Take into account that for the reshaping you may need to use repetitions
(such as [`Tensor.repeat()`](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html))
and other functions, and not only `.view()` operations.

### `reshape_weight` (2.5 points)

This method reshapes the weight tensor created in the init method in a way that
it can be used in the BMM. For this function the target shape is not provided
and there is not an available test. You can figure out if the implementation
is correct in the test for the `forward` pass.

Hint: Take into account that for the reshaping you may need to use repetitions
(such as [`Tensor.repeat()`](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html))
and other functions, and not only `.view()` operations.

### `forward` (2 points)

This method is the complete forward pass. You will have to use the two previous
functions and the BMM to perform the MaxOut.
