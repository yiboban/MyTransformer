from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from grader.adapters import Embedding, Linear, RMSNorm, next_token_cross_entropy, stable_softmax


@pytest.mark.primitives
def test_linear_matches_pytorch_and_has_input_and_weight_gradients():
    layer = Linear(3, 2, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=torch.float64))
    x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    expected = F.linear(x, layer.weight)
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and layer.weight.grad.shape == (2, 3)


@pytest.mark.primitives
def test_linear_zero_weights_produce_zeros_without_bias():
    layer = Linear(2, 3)
    with torch.no_grad():
        layer.weight.zero_()
    assert set(dict(layer.named_parameters())) == {"weight"}
    assert torch.equal(layer(torch.ones(4, 2)), torch.zeros(4, 3))


@pytest.mark.primitives
def test_embedding_supports_arbitrary_leading_dimensions():
    layer = Embedding(7, 3)
    ids = torch.tensor([[[0, 1], [2, 6]], [[3, 4], [5, 0]]])
    expected = F.embedding(ids, layer.weight)
    actual = layer(ids)
    assert actual.shape == (2, 2, 2, 3)
    torch.testing.assert_close(actual, expected)


@pytest.mark.primitives
def test_embedding_rejects_float_ids():
    layer = Embedding(4, 2)
    with pytest.raises((TypeError, ValueError), match="integer|整数|dtype"):
        layer(torch.tensor([1.0]))


@pytest.mark.primitives
def test_rmsnorm_matches_independent_formula_and_preserves_dtype():
    layer = RMSNorm(4, eps=1e-6, dtype=torch.float64)
    x = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], dtype=torch.float64, requires_grad=True)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * layer.weight
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == x.dtype
    actual.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weight.grad).all()


@pytest.mark.primitives
def test_stable_softmax_handles_large_logits_and_sums_to_one():
    x = torch.tensor([[10000.0, 10001.0, 9999.0], [-10000.0, -9999.0, -9998.0]])
    actual = stable_softmax(x, dim=-1)
    torch.testing.assert_close(actual, torch.softmax(x, dim=-1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2))
    assert torch.isfinite(actual).all()


@pytest.mark.primitives
def test_stable_softmax_uses_requested_dimension():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    actual = stable_softmax(x, dim=0)
    torch.testing.assert_close(actual.sum(0), torch.ones(3, dtype=torch.float64))
    assert torch.autograd.grad(actual[0, 0], x)[0].shape == x.shape


@pytest.mark.primitives
def test_next_token_cross_entropy_matches_log_softmax_formula_and_gradients():
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 0, 1]])
    expected = -torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).mean()
    actual = next_token_cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad.shape == logits.shape and torch.isfinite(logits.grad).all()


@pytest.mark.primitives
def test_next_token_cross_entropy_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape|形状"):
        next_token_cross_entropy(torch.randn(2, 3, 5), torch.zeros(2, 2, dtype=torch.long))


@pytest.mark.primitives
def test_primitives_do_not_call_forbidden_high_level_answers():
    forbidden = {
        "layers.py": {"nn.Linear", "F.linear", "nn.Embedding", "F.embedding", "nn.RMSNorm"},
        "functional.py": {"torch.softmax", "F.softmax", "F.cross_entropy"},
    }
    root = Path(__file__).parents[1] / "src" / "mytransformer"
    for filename, blocked in forbidden.items():
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                parts = []
                cursor = node.func
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    parts.append(cursor.id)
                calls.add(".".join(reversed(parts)))
        assert not calls.intersection(blocked), f"{filename} 调用了禁用 API"
