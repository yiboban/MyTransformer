"""Assignment 04: attention 与 multi-head self-attention。"""

import torch
from torch import Tensor, nn

from .layers import Linear
from .rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    # TODO(A04): [B,H,Q,Dh] @ [B,H,K,Dh]，除 sqrt(d_head)，mask 后 stable softmax。
    raise NotImplementedError("Assignment 04: 请实现 scaled_dot_product_attention")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        *,
        use_rope: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # 这些注解只声明公开参数名；真正实例化是学生任务。
        self.q_proj: Linear
        self.k_proj: Linear
        self.v_proj: Linear
        self.o_proj: Linear
        self.rope: RotaryPositionalEmbedding | None
        # TODO(A04): 校验整除，建立独立 Q/K/V/O projection 与可选 RoPE。
        raise NotImplementedError("Assignment 04: 请实现 MultiHeadSelfAttention")

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        raise NotImplementedError("Assignment 04: 请实现 MultiHeadSelfAttention.forward")
