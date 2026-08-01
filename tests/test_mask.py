import pytest
import torch

from grader.adapters import make_causal_mask


@pytest.mark.rope
def test_causal_mask_length_three_exact_values_and_semantics():
    expected = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    actual = make_causal_mask(3)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)


@pytest.mark.rope
def test_causal_mask_rectangular_queries_are_trailing_positions():
    expected = torch.tensor([[True, True, True, False], [True, True, True, True]])
    assert torch.equal(make_causal_mask(2, 4), expected)


@pytest.mark.rope
def test_causal_mask_broadcasts_to_batch_heads():
    mask = make_causal_mask(3)
    scores = torch.zeros(2, 4, 3, 3)
    assert torch.where(mask, scores, torch.tensor(float("-inf"))).shape == scores.shape


@pytest.mark.rope
def test_causal_mask_rejects_invalid_lengths():
    with pytest.raises(ValueError, match="positive|正|length"):
        make_causal_mask(0)
    with pytest.raises(ValueError, match="query|key"):
        make_causal_mask(4, 3)
