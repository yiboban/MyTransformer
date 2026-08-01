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
        # TODO(A03): 校验偶数 head_dim，并缓存/注册旋转频率所需 tensor。
        raise NotImplementedError("Assignment 03: 请实现 RotaryPositionalEmbedding")

    def forward(self, x: Tensor, position_ids: Tensor | None = None) -> Tensor:
        raise NotImplementedError("Assignment 03: 请实现 RoPE.forward")


def make_causal_mask(
    query_length: int,
    key_length: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """返回 [query, key] bool mask，True 表示允许关注。"""
    # TODO(A03): 对非正长度给清晰错误；支持 query 为 key 的末尾窗口。
    raise NotImplementedError("Assignment 03: 请实现 make_causal_mask")
