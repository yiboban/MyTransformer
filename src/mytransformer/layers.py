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
        #  注册形状 [out_features, in_features] 的 weight，无 bias。
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight,std=in_features ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("dim error")
        return x @ self.weight.transpose(0,1)

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
        # 注册 [num_embeddings, embedding_dim] 的 weight。
        # [V,D]
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        # 维度越大 每个分量初始化的越小
        nn.init.trunc_normal_(self.weight,std=embedding_dim ** -0.5)


    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype not in (torch.int32,torch.int64):
            raise ValueError("token_ids must contain integer indices")
        return self.weight[token_ids]

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
        # 注册缩放参数 weight；只沿最后一维计算 RMS。
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.to(dtype=torch.float32)
        fenmu = torch.sqrt(torch.mean(x_float * x_float,dim=-1,keepdim=True)+self.eps)
        fenmu_t = fenmu.to(dtype=self.dtype)
        return x / fenmu_t * self.weight
