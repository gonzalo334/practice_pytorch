import torch

from src.gru_cell import GRUCell


def test_gru_cell_forward_backward() -> None:
    torch.manual_seed(0)
    batch, input_dim, hidden_dim = 4, 5, 6
    inputs = torch.randn(batch, input_dim, dtype=torch.double, requires_grad=True)
    hidden = torch.randn(batch, hidden_dim, dtype=torch.double, requires_grad=True)
    inputs_torch = inputs.detach().clone().requires_grad_(True)
    hidden_torch = hidden.detach().clone().requires_grad_(True)

    model = GRUCell(input_dim, hidden_dim, dtype=torch.double)
    model_torch = torch.nn.GRUCell(input_dim, hidden_dim, dtype=torch.double)
    model.weight_ih = torch.nn.Parameter(model_torch.weight_ih.detach().clone())
    model.weight_hh = torch.nn.Parameter(model_torch.weight_hh.detach().clone())
    model.bias_ih = torch.nn.Parameter(model_torch.bias_ih.detach().clone())
    model.bias_hh = torch.nn.Parameter(model_torch.bias_hh.detach().clone())

    outputs = model(inputs, hidden)
    outputs_torch = model_torch(inputs_torch, hidden_torch)

    assert outputs.shape == outputs_torch.shape
    assert torch.allclose(outputs, outputs_torch, atol=1e-6)

    grad_outputs = torch.randn_like(outputs)
    outputs.backward(grad_outputs)
    outputs_torch.backward(grad_outputs)

    assert torch.allclose(inputs.grad, inputs_torch.grad, atol=1e-6)
    assert torch.allclose(hidden.grad, hidden_torch.grad, atol=1e-6)
    assert torch.allclose(model.weight_ih.grad, model_torch.weight_ih.grad, atol=1e-6)
    assert torch.allclose(model.weight_hh.grad, model_torch.weight_hh.grad, atol=1e-6)
    assert torch.allclose(model.bias_ih.grad, model_torch.bias_ih.grad, atol=1e-6)
    assert torch.allclose(model.bias_hh.grad, model_torch.bias_hh.grad, atol=1e-6)
