import pytest
import torch

from grader.adapters import TransformerBlock, load_weights
from tests.conftest import load_golden


@pytest.mark.block
def test_block_matches_golden_fixture(golden_dir):
    fixture = load_golden(golden_dir / "block_v1.pt")
    block = TransformerBlock(**fixture["config"])
    load_weights(block, fixture["state_dict"])
    actual = block(fixture["input"])
    torch.testing.assert_close(actual, fixture["expected"], atol=1e-6, rtol=1e-5)


@pytest.mark.block
def test_block_is_identity_when_sublayer_output_weights_are_zero():
    block = TransformerBlock(8, 2, 12, 8)
    with torch.no_grad():
        block.attn.o_proj.weight.zero_()
        block.ffn.w2.weight.zero_()
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(block(x), x)


@pytest.mark.block
def test_block_all_parameters_receive_gradients():
    block = TransformerBlock(8, 2, 12, 8)
    x = torch.randn(2, 4, 8, requires_grad=True)
    block(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


@pytest.mark.block
def test_block_is_causal_shortenable_and_batch_independent():
    block = TransformerBlock(8, 2, 12, 8)
    x = torch.randn(2, 6, 8)
    full = block(x)
    torch.testing.assert_close(full[:, :4], block(x[:, :4]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full[0], block(x[:1])[0], atol=1e-6, rtol=1e-5)
