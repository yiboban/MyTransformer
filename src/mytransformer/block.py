"""Assignment 06: pre-norm Transformer block。"""

import torch
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLU
from .layers import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm: RMSNorm
        self.attn: MultiHeadSelfAttention
        self.ffn_norm: RMSNorm
        self.ffn: SwiGLU
        # TODO(A06): 按公开名称构造两个 norm、attention、FFN。
        raise NotImplementedError("Assignment 06: 请实现 TransformerBlock")

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        # TODO(A06): x = x + Attention(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x))。
        raise NotImplementedError("Assignment 06: 请实现 TransformerBlock.forward")
