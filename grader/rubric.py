"""评分阶段的唯一真相源。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    points: int
    marker: str
    docs: str


STAGES = (
    Stage("bpe", "BPE trainer/tokenizer", 20, "bpe", "docs/01_BPE.md"),
    Stage("primitives", "基础层与数值函数", 10, "primitives", "docs/02_基础神经网络模块.md"),
    Stage("rope", "RoPE 与 causal mask", 15, "rope", "docs/03_RoPE与因果Mask.md"),
    Stage("attention", "Attention/MHA", 20, "attention", "docs/04_Attention.md"),
    Stage("ffn", "SiLU/SwiGLU", 10, "ffn", "docs/05_SwiGLU_FFN.md"),
    Stage("block", "TransformerBlock", 10, "block", "docs/06_TransformerBlock.md"),
    Stage("lm", "TransformerLM 与集成", 15, "lm", "docs/07_完整TransformerLM.md"),
)
BY_KEY = {stage.key: stage for stage in STAGES}
