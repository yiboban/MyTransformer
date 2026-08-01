import pytest
import torch

from grader.adapters import TransformerConfig, TransformerLM, next_token_cross_entropy


@pytest.mark.lm
def test_same_seed_reproduces_initialization_and_logits():
    config = TransformerConfig(271, 8, 16, 1, 4, 24)
    torch.manual_seed(123)
    first = TransformerLM(config)
    torch.manual_seed(123)
    second = TransformerLM(config)
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(first(ids), second(ids), rtol=0, atol=0)


@pytest.mark.lm
def test_next_token_labels_are_shifted_by_one_position():
    config = TransformerConfig(32, 8, 16, 1, 4, 24)
    model = TransformerLM(config)
    sequence = torch.tensor([[5, 7, 9, 11, 13]])
    logits = model(sequence[:, :-1])
    loss = next_token_cross_entropy(logits, sequence[:, 1:])
    assert logits.shape == (1, 4, 32) and loss.ndim == 0 and torch.isfinite(loss)
