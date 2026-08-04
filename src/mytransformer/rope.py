"""Assignment 03: Rotary Positional Embedding 与 causal mask。"""

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        # 校验偶数 head_dim，并缓存/注册旋转频率所需 tensor。
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim %2 != 0:
            raise ValueError("head_dim must be even")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        # 每对维度的旋转频率
        # [2]
        dimension_indices = torch.arange(0,head_dim,2,device=device,dtype=torch.float32)
        frequencies = theta ** (-dimension_indices/head_dim) # [2] (x0,x1) 使用频率[0] (x2,x3)使用频率[1]
        #print(omega.shape) # head_dim // 2
        self.register_buffer("frequencies",frequencies)  # buffer 不由优化器更新，但会随模型移动设备并保存到 state_dict。


    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        # x[2,2,3,4] 两个样本 两个head 3个token 每个token4维 [B,H,T,dim]
        # p[3] 3个token的位置是[0,1,2]
        if x.ndim < 2:
            raise ValueError("x must have at least two dimensions")
        if x.shape[-1] != self.head_dim:
            raise ValueError("dimensions must match")
        seq_len = x.shape[-2]
        if position_ids is None:
            if seq_len > self.max_seq_len:
                raise ValueError("length exceed")
            position_ids = torch.arange(0,seq_len,device=x.device)
        if position_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError("position_ids must contain integers")
        if position_ids.ndim not in (1,2):
            raise ValueError("position_ids mast have shape[T] or [B,T]")
        if position_ids.shape[-1] != seq_len:
            raise ValueError("postion_ids last dimension must match sequence length")
        if position_ids.ndim == 2 and position_ids.shape[0] != x.shape[0]:
            raise ValueError("position_ids batch dimension must match x")
        if torch.any(position_ids<0):
            raise ValueError("position_ids must be non-negative")
        if torch.any(position_ids >= self.max_seq_len):
            raise ValueError("exceed max_seq_len")
        position_ids = position_ids.to(device=x.device)
        # 增加一个最后一维 [3,1]
        # 每一个位置要乘所有的频率 每一行表示一个token的两组旋转角度
        angles = position_ids.unsqueeze(-1).to(torch.float32) * self.frequencies
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        if position_ids.ndim == 1:
            # [T,d/2] -> [1,1,T,d/2]
            cos = cos[None,None,:,:]
            sin = sin[None,None,:,:]
        else:
            cos = cos[:,None,:,:]
            sin = sin[:,None,:,:]
        sin = sin.to(dtype=x.dtype)
        cos = cos.to(dtype=x.dtype)
        #print(cos.shape)
        # [2,2,3,2]
        even = x[...,0::2]
        odd = x[...,1::2]
        #print(even.shape)
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        #print(rotated_even.shape)
        output = torch.empty_like(x)
        output[...,0::2] = rotated_even
        output[...,1::2] = rotated_odd
        return output

def make_causal_mask(
    query_length: int,
    key_length: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """返回 [query, key] bool mask，True 表示允许关注。"""
    # 对非正长度给清晰错误；支持 query 为 key 的末尾窗口。
    # 只允许查看自己和过去
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if key_length is None:
        key_length = query_length
    if key_length <= 0:
        raise ValueError("key_length must be positive")
    if query_length > key_length:
        raise ValueError("query_length must not exceed key_length")
    offset = key_length - query_length
    key_position = torch.arange(key_length,device=device)
    query_position = torch.arange(query_length,device=device) + offset
    mask = query_position[:,None] >= key_position[None,:]
    return mask
