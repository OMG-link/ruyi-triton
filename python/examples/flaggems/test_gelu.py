import pytest
import torch

from .gelu import gelu, gelu_, gelu_backward


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_gelu_tanh(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.gelu(x, approximate="tanh")
    tri = gelu(x, approximate="tanh")

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(64, 128), (128, 128)])
@pytest.mark.parametrize("approximate", ["tanh"])
def test_gelu_backward(shape, approximate):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)
    grad_output = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = torch.nn.functional.gelu(x, approximate=approximate)
    ref.backward(grad_output)
    x_grad_ref = x.grad.clone()

    x_grad_tri = gelu_backward(grad_output, x.detach(), approximate=approximate)

    torch.testing.assert_close(x_grad_tri, x_grad_ref, rtol=1e-4, atol=1e-4)


