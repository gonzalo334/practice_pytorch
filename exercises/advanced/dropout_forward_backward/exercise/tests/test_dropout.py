import torch

from src.dropout import Dropout


def test_dropout_eval_identity() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
    model = Dropout(p=0.25)
    model.eval()

    outputs = model(inputs)
    assert torch.allclose(outputs, inputs)

    outputs.sum().backward()
    assert torch.allclose(inputs.grad, torch.ones_like(inputs))


def test_dropout_training_forward_backward() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(8, 6, dtype=torch.double, requires_grad=True)
    model = Dropout(p=0.25)
    model.train()

    outputs = model(inputs)
    scale = 1.0 / (1.0 - model.p)
    kept = outputs != 0
    expected_scale = torch.full_like(outputs[kept], scale)
    assert torch.all((outputs[kept] / inputs[kept]).isclose(expected_scale))

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    assert torch.allclose(inputs.grad, grad_outputs * kept.to(inputs.dtype) * scale)
