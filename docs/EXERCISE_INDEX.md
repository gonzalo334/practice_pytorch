# Exercise Index

This index groups the repository by the concept each exercise practices. Most
folders contain both an `exercise` version and a `solution` version.

## Core Exercises

Core exercises live under `exercises/core/`.

| Folder | Topic | Focus |
| --- | --- | --- |
| `exercises/core/adagrad` | Optimizer | Adagrad state, accumulated squared gradients, parameter updates. |
| `exercises/core/avg_pool2d` | Pooling | Average pooling over 2D inputs. |
| `exercises/core/batchnorm` | Normalization | Custom BatchNorm2d forward and backward behavior. |
| `exercises/core/conv1d` | Convolution | 1D convolution with loop and fold/unfold variants. |
| `exercises/core/conv1d_unfold_fold` | Convolution | 1D convolution expressed through fold/unfold-style operations. |
| `exercises/core/conv2d` | Convolution | Early 2D convolution implementation and tests. |
| `exercises/core/conv2d_forward` | Convolution | Forward pass of Conv2d under loop/vectorization constraints. |
| `exercises/core/custommaxpool` | Pooling | Grouped MaxPool2d that pools across channel groups. |
| `exercises/core/embedding` | Embeddings | Manual backward pass for embedding lookups. |
| `exercises/core/group_norm` | Normalization | GroupNorm forward pass with affine parameters. |
| `exercises/core/group_norm_no_affine` | Normalization | GroupNorm forward pass without affine parameters. |
| `exercises/core/hardshrink` | Activation | Hardshrink forward and backward with masking/indexing. |
| `exercises/core/huber_loss` | Loss | Huber loss forward and backward. |
| `exercises/core/maxout` | Activation | Maxout layer behavior and tensor reshaping. |
| `exercises/core/maxout_forward_backward` | Activation | Maxout forward and backward implementation. |
| `exercises/core/max_pool` | Pooling | Max pooling behavior and test-driven implementation. |
| `exercises/core/nadam` | Optimizer | Nadam optimizer update logic. |
| `exercises/core/parameters_to_double` | Utility | Converting model parameters to double precision. |
| `exercises/core/prelu` | Activation | PReLU module behavior and learnable parameters. |
| `exercises/core/residual` | Models | Residual model components and tests. |
| `exercises/core/softshrink` | Activation | Softshrink forward and backward behavior. |

## Advanced Exercises

Advanced and proposed exercises live under `exercises/advanced/`.

| Folder | Topic | Focus |
| --- | --- | --- |
| `exercises/advanced/adamw` | Optimizer | AdamW update rule and decoupled weight decay. |
| `exercises/advanced/adaptive_avg_pool2d_forward_backward` | Pooling | Adaptive average pooling with backward pass. |
| `exercises/advanced/conv3d_loops_forward_backward` | Convolution | 3D convolution with explicit loops and gradients. |
| `exercises/advanced/conv3d_unfold_fold` | Convolution | 3D convolution using unfold/fold-style tensor operations. |
| `exercises/advanced/convtranspose2d_forward_backward` | Convolution | Transposed 2D convolution forward and backward. |
| `exercises/advanced/cross_entropy_logits_forward_backward` | Loss | Cross entropy directly from logits. |
| `exercises/advanced/custom_grouped_maxpool_forward_backward` | Pooling | Grouped max pooling with manual gradient flow. |
| `exercises/advanced/dropout_forward_backward` | Regularization | Dropout mask behavior in training and evaluation. |
| `exercises/advanced/embedding_bag_forward_backward` | Embeddings | EmbeddingBag aggregation and gradients. |
| `exercises/advanced/embedding_forward_backward` | Embeddings | Embedding lookup forward and backward pass. |
| `exercises/advanced/grouped_conv2d_forward_backward` | Convolution | Grouped Conv2d implementation and gradients. |
| `exercises/advanced/gru_cell_forward_backward` | Sequence models | GRU cell equations and backward pass. |
| `exercises/advanced/instance_norm2d_forward_backward` | Normalization | InstanceNorm2d forward and backward. |
| `exercises/advanced/layernorm_forward_backward` | Normalization | LayerNorm forward and backward. |
| `exercises/advanced/maxout_scatter` | Activation | Maxout implemented with scatter/gather-style gradient routing. |
| `exercises/advanced/rmsprop` | Optimizer | RMSprop moving average and parameter update. |
| `exercises/advanced/rnn_forward_backward` | Sequence models | Vanilla RNN forward and backward pass. |
| `exercises/advanced/scaled_dot_product_attention_forward_backward` | Attention | Scaled dot-product attention and gradients. |

## Reading Suggestions

Good starting points for a reviewer:

1. `exercises/core/maxout` and `exercises/core/maxout_forward_backward` for
   activation design and gradient routing.
2. `exercises/core/conv1d`, `exercises/core/conv2d_forward`, and the advanced
   3D convolution exercises for tensor shape reasoning.
3. `exercises/core/batchnorm`, `exercises/core/group_norm`, and
   `exercises/advanced/layernorm_forward_backward` for normalization mechanics.
4. `exercises/core/adagrad`, `exercises/core/nadam`, and the advanced optimizer
   exercises for training-loop internals.
5. `exercises/advanced/scaled_dot_product_attention_forward_backward` for a
   compact bridge toward transformer-style components.
