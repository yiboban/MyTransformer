from __future__ import annotations

import pytest
import torch

from grader.adapters import TransformerConfig, TransformerLM, load_weights, next_token_cross_entropy
from tests.conftest import load_golden


def tiny_config():
    return TransformerConfig(vocab_size=263, context_length=16, d_model=16, num_layers=2, num_heads=4, d_ff=24)


@pytest.mark.lm
def test_lm_matches_golden_logits(golden_dir):
    fixture = load_golden(golden_dir / "lm_v1.pt")
    model = TransformerLM(TransformerConfig(**fixture["config"]))
    load_weights(model, fixture["state_dict"])
    actual = model(fixture["token_ids"])
    torch.testing.assert_close(actual, fixture["expected"], atol=1e-6, rtol=1e-5)


@pytest.mark.lm
def test_lm_output_shape_context_validation_and_independent_head():
    model = TransformerLM(tiny_config())
    assert model(torch.randint(0, 263, (3, 7))).shape == (3, 7, 263)
    assert model.lm_head.weight is not model.token_embedding.weight
    with pytest.raises(ValueError, match="context|length|长度"):
        model(torch.zeros(1, 17, dtype=torch.long))


@pytest.mark.lm
def test_lm_no_future_leak_and_batch_independence():
    model = TransformerLM(tiny_config()).eval()
    x = torch.randint(0, 263, (2, 8))
    changed = x.clone()
    changed[:, 5:] = (changed[:, 5:] + 17) % 263
    torch.testing.assert_close(model(x)[:, :5], model(changed)[:, :5], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(model(x)[0], model(x[:1])[0], atol=1e-6, rtol=1e-5)


@pytest.mark.lm
def test_lm_state_dict_roundtrip_preserves_output(tmp_path):
    config = tiny_config()
    model = TransformerLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 6))
    expected = model(ids)
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    clone = TransformerLM(config).eval()
    clone.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(clone(ids), expected)


@pytest.mark.lm
def test_lm_backward_loss_finite_and_adamw_changes_parameters():
    config = tiny_config()
    model = TransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, config.vocab_size, (2, 7))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    logits = model(ids[:, :-1])
    loss = next_token_cross_entropy(logits, ids[:, 1:])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())


@pytest.mark.lm
def test_greedy_generation_length_range_and_temperature_validation():
    config = tiny_config()
    model = TransformerLM(config).eval()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    generated = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert generated.shape == (2, 7)
    assert generated.dtype == torch.long
    assert generated.min() >= 0 and generated.max() < config.vocab_size
    with pytest.raises(ValueError, match="temperature|温度"):
        model.generate(prompt, 1, temperature=-1.0)


@pytest.mark.lm
@pytest.mark.slow
def test_tiny_repeated_corpus_loss_decreases():
    torch.manual_seed(7)
    config = TransformerConfig(vocab_size=8, context_length=8, d_model=16, num_layers=1, num_heads=4, d_ff=24)
    model = TransformerLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    data = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2]]).repeat(4, 1)
    with torch.no_grad():
        initial = next_token_cross_entropy(model(data[:, :-1]), data[:, 1:]).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = next_token_cross_entropy(model(data[:, :-1]), data[:, 1:])
        loss.backward()
        optimizer.step()
    final = next_token_cross_entropy(model(data[:, :-1]), data[:, 1:]).item()
    assert final < initial * 0.65
