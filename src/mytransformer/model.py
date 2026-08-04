"""Assignment 07: 完整 Decoder-only Transformer LM。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .block import TransformerBlock
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
        self.token_embedding = Embedding(num_embeddings=config.vocab_size,embedding_dim=config.d_model,device=device,dtype=dtype)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=config.d_model,num_heads=config.num_heads,d_ff=config.d_ff,max_seq_len=config.context_length,rope_theta=config.rope_theta,device=device,dtype=dtype) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(d_model=config.d_model,device=device,dtype=dtype)
        self.lm_head = Linear(in_features=config.d_model,out_features=config.vocab_size,device=device,dtype=dtype)


    def forward(self, token_ids: Tensor) -> Tensor:
        after_token_embedding = self.token_embedding(token_ids)
        after_bolck = after_token_embedding
        for block in self.blocks:
            after_bolck = block(after_bolck)
        after_final_rmsnorm = self.final_norm(after_bolck)
        after_lm_head = self.lm_head(after_final_rmsnorm)
        return after_lm_head # 不需要再softmax 因为next_token_cross_entropy会softmax 接受的是logits

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> Tensor:
        # 截取模型输入窗口
        # → forward
        # → 取最后位置 logits
        # → 选择下一个 token
        # → 拼接到完整序列
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        generated = token_ids.to(dtype=torch.long)
        for _ in range(max_new_tokens):
            model_input = generated[:,-self.config.context_length:]
            logits = self(model_input) # 返回[B,T,V]
            next_logits = logits[:,-1,:]
            if temperature == 0:
                next_token = torch.argmax(next_logits,dim=-1,keepdim=True) # 会对每一个位置都选一个token [B,T,1] 但生成时只需要下一个位置
            else:
                probabilities = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities,num_samples=1)
            generated = torch.cat([generated,next_token],dim=-1)
        return generated
