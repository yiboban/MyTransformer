"""Assignment 04: attention 与 multi-head self-attention。"""

import torch
from torch import Tensor, nn

from .layers import Linear
from .rope import RotaryPositionalEmbedding, make_causal_mask


def scaled_dot_product_attention(
    q: Tensor, # [B,H,Q,Dh] Q表示查询位置
    k: Tensor, # [B,H,K,Dh] K表示被查询位置
    v: Tensor, # [B,H,K,Dh]
    mask: Tensor | None = None,
) -> Tensor:
    if q.ndim != 4:
        raise ValueError("q must have shape [B, H, Q, Dh]")
    if k.ndim != 4:
        raise ValueError("k must have shape [B, H, K, Dh]")
    if v.ndim != 4:
        raise ValueError("v must have shape [B, H, K, Dh]")
    # batch 和 head 数必须一致
    if q.shape[:2] != k.shape[:2]:
        raise ValueError("q and k batch/head dimensions must match")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError("q and v batch/head dimensions must match")
    # Q 和 K 的点积维度必须一致
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions must match")
    # 每个 Key 必须有对应的 Value
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v sequence lengths must match")
    d = q.shape[-1]
    scores = q @ k.transpose(-1,-2) / (d ** 0.5)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("mask must have bool dtype")
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        elif mask.ndim != 4:
            raise ValueError("mask must have 2, 3, or 4 dimensions")
        if mask.shape[-2:] != scores.shape[-2:]:
            raise ValueError("mask query/key dimensions must match scores")
        mask = mask.to(device=scores.device)
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("each query row must allow at least one key")
        scores = scores.masked_fill(~mask,float("-inf"))
    scores = torch.softmax(scores,dim=-1)
    return scores @ v


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
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.q_proj = Linear( d_model, d_model, device=device,dtype=dtype,)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype,)
        self.v_proj = Linear(d_model, d_model,device=device,dtype=dtype)
        self.o_proj = Linear( d_model,d_model,device=device,dtype=dtype,)
        if use_rope:
            # 传x 和position_ids
            self.rope = RotaryPositionalEmbedding(
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
                theta=rope_theta,
                device=device,
            )
        else:
            self.rope = None

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        # x [B,T,D]
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x last dimension must be {self.d_model}, "f"but got {x.shape[-1]}")
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds "f"max_seq_len {self.max_seq_len}")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        q = q.transpose(1,2)
        v = v.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        v = v.transpose(1,2)
        k = k.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        k = k.transpose(1,2)
        if self.rope is not None:
            q = self.rope(q,position_ids)
            k = self.rope(k,position_ids)
        # 最关键 causal mask创建
        mask = make_causal_mask(seq_len,seq_len,device=x.device)
        attn_output = scaled_dot_product_attention(q,k,v,mask)
        #print(attn_output.shape)
        # 归位 传proj_o
        attn_output = attn_output.transpose(1,2)
        attn_output = attn_output.reshape(batch_size,seq_len,_)
        return self.o_proj(attn_output)
