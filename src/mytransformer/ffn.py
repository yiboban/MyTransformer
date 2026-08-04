"""Assignment 05: SiLU 与 SwiGLU。"""

import torch
from torch import Tensor, nn

from .layers import Linear


# silu(x) = x * sigmoid(x) 保证在0附近也是平滑的 没有ReLU的尖角
# sigmoid(x) = 1 / (1+e^-x)
def silu(x: Tensor) -> Tensor:
    # 使用 x * sigmoid(x)，不要调用高层 SiLU 答案。
    return x * torch.sigmoid(x)

# SwiGLU(x) = W2(SiLU(W1x)*W3x)
# W1是产生经过SiLU的门控分支
# W3是产生线性内容分支
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # 三个无 bias Linear；名称与公式 W1/W2/W3 对齐。
        # x : [B,T,d_model]
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
