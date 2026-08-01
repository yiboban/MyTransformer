"""测试与学生实现之间的薄 adapter。

只允许导入、实例化和明确的权重装载；禁止在这里实现算法。
若学生更改模块路径，只需修改本文件。
"""

from importlib import import_module
from inspect import signature
from typing import Any


class AdapterError(RuntimeError):
    """学生 API 无法导入或不满足契约。"""


def _symbol(module_name: str, name: str) -> Any:
    try:
        module = import_module(module_name)
    except Exception as exc:
        raise AdapterError(
            f"无法导入 {module_name}。请确认 src 布局和安装步骤；原始错误: {exc}"
        ) from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AdapterError(f"{module_name} 缺少公开 API {name}") from exc


def _require_parameters(obj: Any, display_name: str, required: tuple[str, ...]) -> Any:
    """导入时只校验公开参数名；不适配或补做学生算法。"""
    try:
        available = set(signature(obj).parameters)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"无法读取 {display_name} 的函数签名: {exc}") from exc
    missing = [name for name in required if name not in available]
    if missing:
        raise AdapterError(f"{display_name} 签名缺少参数: {', '.join(missing)}")
    return obj


train_bpe = _symbol("mytransformer.tokenizer", "train_bpe")
BPETokenizer = _symbol("mytransformer.tokenizer", "BPETokenizer")
Linear = _symbol("mytransformer.layers", "Linear")
Embedding = _symbol("mytransformer.layers", "Embedding")
RMSNorm = _symbol("mytransformer.layers", "RMSNorm")
stable_softmax = _symbol("mytransformer.functional", "stable_softmax")
next_token_cross_entropy = _symbol("mytransformer.functional", "next_token_cross_entropy")
RotaryPositionalEmbedding = _symbol("mytransformer.rope", "RotaryPositionalEmbedding")
make_causal_mask = _symbol("mytransformer.rope", "make_causal_mask")
scaled_dot_product_attention = _symbol("mytransformer.attention", "scaled_dot_product_attention")
MultiHeadSelfAttention = _symbol("mytransformer.attention", "MultiHeadSelfAttention")
silu = _symbol("mytransformer.ffn", "silu")
SwiGLU = _symbol("mytransformer.ffn", "SwiGLU")
TransformerBlock = _symbol("mytransformer.block", "TransformerBlock")
TransformerConfig = _symbol("mytransformer.model", "TransformerConfig")
TransformerLM = _symbol("mytransformer.model", "TransformerLM")

_require_parameters(train_bpe, "train_bpe", ("input_path", "vocab_size", "special_tokens"))
_require_parameters(BPETokenizer, "BPETokenizer", ("vocab", "merges", "special_tokens"))
_require_parameters(scaled_dot_product_attention, "scaled_dot_product_attention", ("q", "k", "v", "mask"))
_require_parameters(MultiHeadSelfAttention, "MultiHeadSelfAttention", ("d_model", "num_heads"))
_require_parameters(TransformerLM, "TransformerLM", ("config",))


def load_weights(module: Any, state_dict: dict[str, Any]) -> None:
    """严格装载 fixture 权重，参数布局不一致时保留 PyTorch 原始错误。"""
    module.load_state_dict(state_dict, strict=True)
