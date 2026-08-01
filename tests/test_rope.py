from __future__ import annotations

import pytest
import torch

from grader.adapters import RotaryPositionalEmbedding
from tests.conftest import load_golden


@pytest.mark.rope
def test_rope_position_zero_is_identity():
    rope = RotaryPositionalEmbedding(4, 8)
    x = torch.randn(2, 3, 1, 4)
    positions = torch.zeros(2, 1, dtype=torch.long)
    torch.testing.assert_close(rope(x, positions), x)


@pytest.mark.rope
def test_rope_matches_small_golden_fixture(golden_dir):
    fixture = load_golden(golden_dir / "rope_v1.pt")
    rope = RotaryPositionalEmbedding(**fixture["config"])
    actual = rope(fixture["input"], fixture["position_ids"])
    torch.testing.assert_close(actual, fixture["expected"], atol=1e-6, rtol=1e-6)


@pytest.mark.rope
def test_rope_supports_batched_explicit_positions_and_gradients():
    rope = RotaryPositionalEmbedding(6, 16)
    x = torch.randn(2, 3, 4, 6, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = rope(x, positions)
    assert y.shape == x.shape and y.dtype == x.dtype
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.rope
def test_rope_preserves_each_two_dimensional_pair_norm():
    rope = RotaryPositionalEmbedding(4, 8)
    x = torch.randn(1, 2, 5, 4)
    y = rope(x)
    torch.testing.assert_close(x.reshape(*x.shape[:-1], 2, 2).norm(dim=-1), y.reshape(*y.shape[:-1], 2, 2).norm(dim=-1))


@pytest.mark.rope
def test_rope_rejects_odd_dimension_and_out_of_range_position():
    with pytest.raises(ValueError, match="even|偶"):
        RotaryPositionalEmbedding(3, 8)
    rope = RotaryPositionalEmbedding(4, 2)
    with pytest.raises(ValueError, match="max|length|范围"):
        rope(torch.randn(1, 1, 3, 4))
