"""Assignment 07: 完整 Decoder-only Transformer LM。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import Embedding, Linear, RMSNorm


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"配置项必须为正数: {', '.join(bad)}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE 要求 head dimension 为偶数")


class TransformerLM(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.token_embedding: Embedding
        self.blocks: nn.ModuleList
        self.final_norm: RMSNorm
        self.lm_head: Linear
        # TODO(A07): 独立 lm_head（默认不绑 embedding），无 dropout。
        raise NotImplementedError("Assignment 07: 请实现 TransformerLM")

    def forward(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError("Assignment 07: 请实现 TransformerLM.forward")

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> Tensor:
        raise NotImplementedError("Assignment 07: 请实现 TransformerLM.generate")
