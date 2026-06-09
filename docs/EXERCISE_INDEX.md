# Exercise Index

This index groups the repository by the concept each exercise practices. Most
folders contain both an `exercise` version and a `solution` version.

## Core Exercises

| Folder | Topic | Focus |
| --- | --- | --- |
| `adagrad` | Optimizer | Adagrad state, accumulated squared gradients, parameter updates. |
| `avg_pool2d` | Pooling | Average pooling over 2D inputs. |
| `batchnorm` | Normalization | Custom BatchNorm2d forward and backward behavior. |
| `conv1d` | Convolution | 1D convolution with loop and fold/unfold variants. |
| `conv1d_unfold_fold` | Convolution | 1D convolution expressed through fold/unfold-style operations. |
| `conv2d` | Convolution | Early 2D convolution implementation and tests. |
| `conv2d_forward` | Convolution | Forward pass of Conv2d under loop/vectorization constraints. |
| `custommaxpool` | Pooling | Grouped MaxPool2d that pools across channel groups. |
| `embedding` | Embeddings | Manual backward pass for embedding lookups. |
| `group_norm` | Normalization | GroupNorm forward pass with affine parameters. |
| `group_norm_no_affine` | Normalization | GroupNorm forward pass without affine parameters. |
| `hardshrink` | Activation | Hardshrink forward and backward with masking/indexing. |
| `huber_loss` | Loss | Huber loss forward and backward. |
| `maxout` | Activation | Maxout layer behavior and tensor reshaping. |
| `maxout_forward_backward` | Activation | Maxout forward and backward implementation. |
| `max_pool` | Pooling | Max pooling behavior and test-driven implementation. |
| `nadam` | Optimizer | Nadam optimizer update logic. |
| `parameters_to_double` | Utility | Converting model parameters to double precision. |
| `prelu` | Activation | PReLU module behavior and learnable parameters. |
| `residual` | Models | Residual model components and tests. |
| `softshrink` | Activation | Softshrink forward and backward behavior. |

## Proposed And Extended Exercises

| Folder | Topic | Focus |
| --- | --- | --- |
| `_proposed_exercises/adamw` | Optimizer | AdamW update rule and decoupled weight decay. |
| `_proposed_exercises/adaptive_avg_pool2d_forward_backward` | Pooling | Adaptive average pooling with backward pass. |
| `_proposed_exercises/conv3d_loops_forward_backward` | Convolution | 3D convolution with explicit loops and gradients. |
| `_proposed_exercises/conv3d_unfold_fold` | Convolution | 3D convolution using unfold/fold-style tensor operations. |
| `_proposed_exercises/convtranspose2d_forward_backward` | Convolution | Transposed 2D convolution forward and backward. |
| `_proposed_exercises/cross_entropy_logits_forward_backward` | Loss | Cross entropy directly from logits. |
| `_proposed_exercises/custom_grouped_maxpool_forward_backward` | Pooling | Grouped max pooling with manual gradient flow. |
| `_proposed_exercises/dropout_forward_backward` | Regularization | Dropout mask behavior in training and evaluation. |
| `_proposed_exercises/embedding_bag_forward_backward` | Embeddings | EmbeddingBag aggregation and gradients. |
| `_proposed_exercises/embedding_forward_backward` | Embeddings | Embedding lookup forward and backward pass. |
| `_proposed_exercises/grouped_conv2d_forward_backward` | Convolution | Grouped Conv2d implementation and gradients. |
| `_proposed_exercises/gru_cell_forward_backward` | Sequence models | GRU cell equations and backward pass. |
| `_proposed_exercises/instance_norm2d_forward_backward` | Normalization | InstanceNorm2d forward and backward. |
| `_proposed_exercises/layernorm_forward_backward` | Normalization | LayerNorm forward and backward. |
| `_proposed_exercises/maxout_scatter` | Activation | Maxout implemented with scatter/gather-style gradient routing. |
| `_proposed_exercises/rmsprop` | Optimizer | RMSprop moving average and parameter update. |
| `_proposed_exercises/rnn_forward_backward` | Sequence models | Vanilla RNN forward and backward pass. |
| `_proposed_exercises/scaled_dot_product_attention_forward_backward` | Attention | Scaled dot-product attention and gradients. |

## Reading Suggestions

Good starting points for a reviewer:

1. `maxout/` and `maxout_forward_backward/` for activation design and gradient
   routing.
2. `conv1d/`, `conv2d_forward/`, and the proposed 3D convolution exercises for
   tensor shape reasoning.
3. `batchnorm/`, `group_norm/`, and proposed `layernorm_forward_backward/` for
   normalization mechanics.
4. `adagrad/`, `nadam/`, and proposed optimizer exercises for training-loop
   internals.
5. `_proposed_exercises/scaled_dot_product_attention_forward_backward/` for a
   compact bridge toward transformer-style components.
