# PyTorch Internals Practice

> A portfolio of Deep Learning implementation drills: layers, losses,
> optimizers, manual gradients, and tensor mechanics rebuilt from first
> principles in PyTorch.

This repository is a record of deliberate practice. Instead of treating PyTorch
as a black box, these exercises rebuild the pieces that make neural networks
work: convolution windows, pooling indices, normalization statistics, activation
masks, embedding gradients, recurrent state updates, attention weights, and
optimizer moments.

The result is not a single application. It is a workshop of compact,
test-driven implementations designed to show low-level Deep Learning fluency:
how tensors move, how gradients return, and how training algorithms update
parameters over time.

## At A Glance

| Area | Examples |
| --- | --- |
| Convolutions and pooling | `conv1d`, `conv2d_forward`, `conv3d`, grouped convolution, transposed convolution, max/average/adaptive pooling |
| Normalization | BatchNorm, GroupNorm, InstanceNorm, LayerNorm |
| Activations | Maxout, PReLU, Hardshrink, Softshrink |
| Losses | Huber loss, cross-entropy from logits |
| Embeddings and sequence models | Embedding, EmbeddingBag, RNN, GRU, scaled dot-product attention |
| Optimizers | Adagrad, Nadam, RMSprop, AdamW |

## What This Demonstrates

- Reimplementing neural network components without leaning on high-level
  `torch.nn` shortcuts.
- Matching PyTorch semantics through focused unit tests.
- Reasoning carefully about tensor shapes, broadcasting, grouping, indexing, and
  memory layout.
- Writing manual backward passes for differentiable operators.
- Comparing direct loop-based implementations with vectorized tensor programs.
- Building optimizer update rules from their mathematical definitions.

## Repository Structure

Most folders are self-contained exercises. The common structure is:

```text
exercises/
  core/                 # Original practice set
    exercise_name/
      exercise/
        src/            # Implementation target
        tests/          # Pytest tests for the exercise
        README.md       # Problem statement and constraints
      solution/
        src/            # Reference or completed solution
        tests/          # Matching tests
        README.md
  advanced/             # Extended and proposed exercises
    exercise_name/
      exercise/
      solution/
```

Some earlier exercises use a flatter layout:

```text
exercises/core/exercise_name/
  src/
  tests/
```

The root stays intentionally small: documentation, Git metadata, and the
organized exercise tree.

## Exercise Index

For a categorized index of the exercises, see
[docs/EXERCISE_INDEX.md](docs/EXERCISE_INDEX.md).

Representative areas covered in this repo:

- **Convolutions and pooling:** `conv1d`, `conv2d`, `conv3d`, transposed
  convolution, grouped convolution, max/average/adaptive pooling.
- **Normalization and activations:** BatchNorm, GroupNorm, InstanceNorm,
  LayerNorm, PReLU, Hardshrink, Softshrink, Maxout.
- **Losses:** Huber loss and cross-entropy from logits.
- **Embeddings and sequence models:** Embedding, EmbeddingBag, RNN cell, GRU
  cell, scaled dot-product attention.
- **Optimizers and utilities:** Adagrad, Nadam, RMSprop, AdamW, parameter dtype
  conversion.
- **Residual models:** simple residual building blocks and model tests.

## Why These Exercises Matter

Modern Deep Learning code often compresses powerful ideas into one-line module
calls. That is productive, but it can hide the machinery:

- a convolution is a structured set of overlapping tensor reads;
- max pooling is both a reduction and a routing problem for gradients;
- normalization layers are statistics, shape transforms, and learned affine
  parameters living together;
- embeddings look simple in the forward pass but require scatter-style gradient
  accumulation;
- optimizers are stateful algorithms, not just calls to `step()`.

These exercises make those mechanics explicit. They are small enough to inspect,
but broad enough to cover many of the building blocks behind CNNs, sequence
models, and transformer-style components.

## How To Run An Exercise

Each exercise is intentionally self-contained. From an exercise folder, install
the listed requirements and run the tests.

Example:

```bash
cd exercises/core/maxout/exercise
pip install -r requirements.txt
pytest
```

Some folders include a `test.sh` helper:

```bash
cd exercises/core/adagrad/exercise
bash test.sh
```

On Windows PowerShell, running `pytest` directly from the exercise directory is
usually the most convenient path.

## Suggested Reading Path

If you are reviewing this as part of a portfolio, these folders give a good tour:

1. `exercises/core/maxout/` and `exercises/core/maxout_forward_backward/` for
   activation design and gradient routing.
2. `exercises/core/conv1d/`, `exercises/core/conv2d_forward/`, and
   `exercises/advanced/conv3d_*` for tensor shape reasoning.
3. `exercises/core/batchnorm/`, `exercises/core/group_norm/`, and
   `exercises/advanced/layernorm_forward_backward/` for normalization mechanics.
4. `exercises/core/adagrad/`, `exercises/core/nadam/`,
   `exercises/advanced/rmsprop/`, and `exercises/advanced/adamw/` for optimizer
   internals.
5. `exercises/advanced/scaled_dot_product_attention_forward_backward/` for a
   compact bridge toward transformer components.

## Development Notes

- The code is organized as learning artifacts rather than a shared Python
  package.
- Tests are the main contract for each implementation.
- Many READMEs include constraints such as "no `nn` package", "no loops", or
  "no `torch.where`"; those constraints are part of the learning objective.
- Generated files such as virtual environments, pytest caches, mypy caches, and
  bytecode are ignored and should not be committed.

## Why This Repository Exists

Deep Learning libraries are powerful, but they can hide the mechanics that make
models train. These exercises were a way to slow down and rebuild the pieces:
how windows slide across tensors, how gradients scatter back into inputs, how
normalization statistics are shaped, and how optimizers update parameters over
time.

The result is a practical notebook of implementation work that complements
larger projects by showing low-level understanding of PyTorch and neural network
building blocks.
