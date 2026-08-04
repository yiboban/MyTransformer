"""Assignment 06: pre-norm Transformer block。"""

import torch
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLU
from .layers import RMSNorm


# 开始组合
# h = x + attention(rmsnorm(x))
# y = h + swiglu(rmsnorm(h))
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
        # 按公开名称构造两个 norm、attention、FFN。
        self.attn_norm = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,rope_theta=rope_theta,device=device,dtype=dtype)
        self.ffn_norm = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        # x = x + Attention(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x))。
        x = x + self.attn(self.attn_norm(x),position_ids)
        return x + self.ffn(self.ffn_norm(x))
