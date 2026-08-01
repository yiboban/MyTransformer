"""Assignment 05: SiLU 与 SwiGLU。"""

import torch
from torch import Tensor, nn

from .layers import Linear


def silu(x: Tensor) -> Tensor:
    # TODO(A05): 使用 x * sigmoid(x)，不要调用高层 SiLU 答案。
    raise NotImplementedError("Assignment 05: 请实现 silu")


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
        self.w1: Linear
        self.w2: Linear
        self.w3: Linear
        # TODO(A05): 三个无 bias Linear；名称与公式 W1/W2/W3 对齐。
        raise NotImplementedError("Assignment 05: 请实现 SwiGLU")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Assignment 05: 请实现 SwiGLU.forward")
