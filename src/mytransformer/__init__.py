"""学生需要逐步完成的 Decoder-only Transformer 包。"""

from .attention import MultiHeadSelfAttention, scaled_dot_product_attention
from .block import TransformerBlock
from .ffn import SwiGLU, silu
from .functional import next_token_cross_entropy, stable_softmax
from .layers import Embedding, Linear, RMSNorm
from .model import TransformerConfig, TransformerLM
from .rope import RotaryPositionalEmbedding, make_causal_mask
from .tokenizer import BPETokenizer, train_bpe

__all__ = [
    "BPETokenizer",
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerConfig",
    "TransformerLM",
    "make_causal_mask",
    "next_token_cross_entropy",
    "scaled_dot_product_attention",
    "silu",
    "stable_softmax",
    "train_bpe",
]
