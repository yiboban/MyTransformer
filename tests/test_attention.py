from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from grader.adapters import MultiHeadSelfAttention, load_weights, scaled_dot_product_attention
from tests.conftest import load_golden


@pytest.mark.attention
def test_zero_queries_and_keys_produce_uniform_average():
    q = torch.zeros(1, 2, 3, 4)
    k = torch.zeros(1, 2, 3, 4)
    v = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    expected = v.mean(dim=-2, keepdim=True).expand_as(v)
    torch.testing.assert_close(scaled_dot_product_attention(q, k, v), expected)


@pytest.mark.attention
def test_single_token_attention_returns_value():
    q, k, v = (torch.randn(2, 3, 1, 4) for _ in range(3))
    torch.testing.assert_close(scaled_dot_product_attention(q, k, v), v)


@pytest.mark.attention
def test_attention_matches_pytorch_reference_with_allow_mask_and_gradients():
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    actual = scaled_dot_product_attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
    actual.sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.attention
def test_masked_position_receives_exactly_zero_probability_effect():
    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 1, 2, 2)
    v = torch.tensor([[[[2.0, 4.0], [1000.0, 1000.0]]]])
    mask = torch.tensor([[True, False]])
    assert torch.equal(scaled_dot_product_attention(q, k, v, mask), v[..., :1, :])


@pytest.mark.attention
def test_attention_scaling_uses_head_dimension():
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = q.clone()
    v = torch.tensor([[[[1.0], [3.0]]]])
    scores = q @ k.transpose(-1, -2) / math.sqrt(2)
    expected = torch.softmax(scores, -1) @ v
    torch.testing.assert_close(scaled_dot_product_attention(q, k, v), expected)


@pytest.mark.attention
def test_mha_without_rope_matches_torch_multihead_attention():
    ours = MultiHeadSelfAttention(8, 2, 8, use_rope=False)
    reference = nn.MultiheadAttention(8, 2, dropout=0.0, bias=False, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight], dim=0))
        reference.out_proj.weight.copy_(ours.o_proj.weight)
    reference.eval()
    x = torch.randn(2, 4, 8)
    allow = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    actual = ours(x)
    expected, _ = reference(x, x, x, attn_mask=~allow, need_weights=False)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.attention
def test_mha_with_rope_matches_golden_fixture(golden_dir):
    fixture = load_golden(golden_dir / "mha_rope_v1.pt")
    module = MultiHeadSelfAttention(**fixture["config"])
    load_weights(module, fixture["state_dict"])
    actual = module(fixture["input"], fixture["position_ids"])
    torch.testing.assert_close(actual, fixture["expected"], atol=1e-6, rtol=1e-5)


@pytest.mark.attention
def test_mha_is_causal_and_batch_independent():
    module = MultiHeadSelfAttention(8, 2, 8)
    prefix = torch.randn(1, 3, 8)
    x1 = torch.cat([prefix, torch.randn(1, 2, 8)], dim=1)
    x2 = torch.cat([prefix, torch.randn(1, 2, 8) * 10], dim=1)
    torch.testing.assert_close(module(x1)[:, :3], module(x2)[:, :3], atol=1e-6, rtol=1e-5)
    batch = torch.cat([x1, x2], dim=0)
    torch.testing.assert_close(module(batch)[0], module(x1)[0], atol=1e-6, rtol=1e-5)


@pytest.mark.attention
def test_mha_rejects_non_divisible_model_dimension():
    with pytest.raises(ValueError, match="divisible|整除"):
        MultiHeadSelfAttention(7, 2, 8)


@pytest.mark.attention
def test_attention_student_code_avoids_fused_or_transformer_answers():
    blocked = {"F.scaled_dot_product_attention", "nn.MultiheadAttention", "nn.Transformer", "nn.TransformerEncoder", "nn.TransformerDecoder"}
    tree = ast.parse((Path(__file__).parents[1] / "src/mytransformer/attention.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                found.add(ast.unparse(node.func))
            except AttributeError:
                pass
    assert not found.intersection(blocked)
