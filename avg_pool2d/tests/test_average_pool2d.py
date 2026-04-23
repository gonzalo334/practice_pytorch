import torch
import pytest

from src.average_pool2d import AveragePool2d


def test_output_shape():
    x = torch.randn(1, 1, 4, 4)

    layer = AveragePool2d(kernel_size=2, stride=2)
    y = layer(x)

    assert y.shape == (1, 1, 2, 2)


def test_basic_case():
    x = torch.tensor(
        [[[[1.0, 2.0, 3.0, 0.0],
           [4.0, 5.0, 6.0, 1.0],
           [7.0, 8.0, 9.0, 2.0],
           [0.0, 1.0, 2.0, 3.0]]]]
    )

    layer = AveragePool2d(kernel_size=2, stride=2)
    y = layer(x)

    expected = torch.tensor(
        [[[[3.0, 2.5],
           [4.0, 4.0]]]]
    )

    assert torch.allclose(y, expected)


def test_padding():
    x = torch.ones(1, 1, 3, 3)

    layer = AveragePool2d(kernel_size=2, stride=1, padding=1)
    y = layer(x)

    # Only check shape (values depend on padding implementation)
    assert y.shape == (1, 1, 4, 4)


def test_multi_channel():
    x = torch.randn(2, 3, 6, 6)

    layer = AveragePool2d(kernel_size=2, stride=2)
    y = layer(x)

    assert y.shape == (2, 3, 3, 3)


def test_compare_with_pytorch():
    x = torch.randn(1, 2, 8, 8)

    custom = AveragePool2d(kernel_size=2, stride=2)
    torch_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    y1 = custom(x)
    y2 = torch_layer(x)

    assert torch.allclose(y1, y2, atol=1e-5)