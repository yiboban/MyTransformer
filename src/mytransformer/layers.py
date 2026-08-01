"""Assignment 02: Linear、Embedding、RMSNorm 学生骨架。"""

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # TODO(A02): 注册形状 [out_features, in_features] 的 weight，无 bias。
        raise NotImplementedError("Assignment 02: 请实现 Linear")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Assignment 02: 请实现 Linear.forward")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # TODO(A02): 注册 [num_embeddings, embedding_dim] 的 weight。
        raise NotImplementedError("Assignment 02: 请实现 Embedding")

    def forward(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError("Assignment 02: 请实现 Embedding.forward")


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # TODO(A02): 注册缩放参数 weight；只沿最后一维计算 RMS。
        raise NotImplementedError("Assignment 02: 请实现 RMSNorm")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Assignment 02: 请实现 RMSNorm.forward")
