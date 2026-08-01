import pytest
import torch
import torch.nn.functional as F

from grader.adapters import SwiGLU, silu


@pytest.mark.ffn
def test_silu_matches_pytorch_and_has_finite_gradient():
    x = torch.linspace(-3, 3, 13, dtype=torch.float64, requires_grad=True)
    actual = silu(x)
    torch.testing.assert_close(actual, F.silu(x))
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.ffn
def test_swiglu_matches_explicit_formula_and_shapes():
    module = SwiGLU(4, 7, dtype=torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    expected = module.w2(F.silu(module.w1(x)) * module.w3(x))
    actual = module(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.ffn
def test_swiglu_all_parameters_receive_finite_gradients():
    module = SwiGLU(6, 10)
    x = torch.randn(3, 2, 6, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None
    assert set(dict(module.named_parameters())) == {"w1.weight", "w2.weight", "w3.weight"}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.ffn
@pytest.mark.parametrize("shape", [(1, 1, 4), (3, 5, 4)])
def test_swiglu_supports_arbitrary_batch_and_sequence(shape):
    assert SwiGLU(4, 8)(torch.randn(*shape)).shape == shape
