from __future__ import annotations

import ast
import inspect

import grader.adapters as adapters


def test_adapter_exports_every_public_contract():
    required = {
        "train_bpe",
        "BPETokenizer",
        "Linear",
        "Embedding",
        "RMSNorm",
        "stable_softmax",
        "next_token_cross_entropy",
        "RotaryPositionalEmbedding",
        "make_causal_mask",
        "scaled_dot_product_attention",
        "MultiHeadSelfAttention",
        "silu",
        "SwiGLU",
        "TransformerBlock",
        "TransformerConfig",
        "TransformerLM",
        "load_weights",
    }
    assert required.issubset(vars(adapters))


def test_adapter_remains_thin_and_contains_no_tensor_algorithm():
    source = inspect.getsource(adapters)
    tree = ast.parse(source)
    forbidden_calls = {"matmul", "einsum", "softmax", "exp", "rsqrt", "tril", "sigmoid"}
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called.intersection(forbidden_calls)
    assert len(source.splitlines()) < 100


def test_adapter_load_weights_is_strict():
    tree = ast.parse(inspect.getsource(adapters.load_weights))
    keywords = {
        kw.arg: kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if isinstance(kw.value, ast.Constant)
    }
    assert keywords.get("strict") is True
