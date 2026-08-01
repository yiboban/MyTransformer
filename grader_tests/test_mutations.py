"""用独立的小 oracle 证明 grader 能击中常见错误，而不污染学生代码。"""

from __future__ import annotations

import math

import pytest
import torch


def correct_attention(q, k, v, mask):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, -1) @ v


def assert_attention_oracle_rejects(candidate):
    q = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    k = torch.tensor([[[[1.0, 1.0], [2.0, 0.0]]]])
    v = torch.tensor([[[[1.0, 2.0], [20.0, 30.0]]]])
    mask = torch.tensor([[True, False], [True, True]])
    expected = correct_attention(q, k, v, mask)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(candidate(q, k, v, mask), expected, atol=1e-6, rtol=1e-6)


def test_rejects_attention_scaled_by_sqrt_d_model():
    def mutant(q, k, v, mask):
        scores = q @ k.transpose(-1, -2) / math.sqrt(8)
        return torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ v

    assert_attention_oracle_rejects(mutant)


def test_rejects_softmax_wrong_dimension():
    def mutant(q, k, v, mask):
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        return torch.softmax(scores.masked_fill(~mask, float("-inf")), -2) @ v

    assert_attention_oracle_rejects(mutant)


def test_rejects_mask_after_softmax_and_reversed_bool_semantics():
    def after(q, k, v, mask):
        probabilities = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
        return (probabilities * mask) @ v

    def reversed_mask(q, k, v, mask):
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        return torch.softmax(scores.masked_fill(mask, float("-inf")), -1) @ v

    assert_attention_oracle_rejects(after)
    assert_attention_oracle_rejects(reversed_mask)


def test_rejects_qk_transpose_error():
    def mutant(q, k, v, mask):
        scores = q.transpose(-1, -2) @ k / math.sqrt(q.shape[-1])
        return torch.softmax(scores, -1) @ v

    assert_attention_oracle_rejects(mutant)


def rotate_pairs(x):
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


def test_rejects_rope_applied_to_values():
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    expected = v
    mutated = rotate_pairs(v)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(mutated, expected)


def test_rejects_unicode_character_bpe_and_nondeterministic_tie_break():
    text = "你a"
    assert len(list(text)) != len(text.encode("utf-8"))
    tied = {(b"c", b"d"): 1, (b"a", b"b"): 1}
    deterministic = min(tied, key=lambda pair: (-tied[pair], pair))
    insertion_dependent = next(iter(tied))
    assert deterministic != insertion_dependent


def test_rejects_swiglu_without_w3_branch():
    x = torch.tensor([[-1.0, 2.0]])
    w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    w3 = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    correct = torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)
    mutant = torch.nn.functional.silu(x @ w1.T)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(mutant, correct)


def test_rejects_missing_residual_and_post_norm():
    x = torch.tensor([[1.0, 2.0]])
    sublayer = torch.tensor([[0.5, -0.5]])
    assert not torch.equal(sublayer, x + sublayer)
    pre = x + (x / x.square().mean(-1, keepdim=True).sqrt())
    post_input = x + x
    post = post_input / post_input.square().mean(-1, keepdim=True).sqrt()
    assert not torch.allclose(pre, post)


def test_rejects_complete_lm_future_leak():
    def acausal(ids):
        return ids.float().sum(-1, keepdim=True).unsqueeze(1).expand(-1, ids.shape[1], -1)

    original = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 9, 9]])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(acausal(original)[:, :2], acausal(changed)[:, :2])
