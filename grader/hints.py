"""面向初学者的失败分类，不隐藏 pytest 原始信息。"""

from .rubric import Stage

HINTS = {
    "bpe": "检查 UTF-8 bytes、merge rank、确定性 tie-breaking 和 special token longest match。",
    "primitives": "检查最后一维、weight 形状、减最大值，以及 dtype/device 是否随输入传播。",
    "rope": "检查偶数维两两旋转、position ids，以及 bool mask 中 True=允许。",
    "attention": (
        "Attention 失败时请检查：1) 除以 sqrt(d_head)；2) mask 在 softmax 前应用；"
        "3) True 表示允许；4) mask 广播到 [B,H,Q,K]；5) Q/K 的 transpose。"
    ),
    "ffn": "检查 SiLU(W1x) 与 W3x 逐元素相乘后再经过 W2，三个 Linear 均无 bias。",
    "block": "检查 pre-norm 顺序和两个 residual：每个子层都要把结果加回原 x。",
    "lm": "检查 context_length、因果性、独立 lm_head、state_dict 和 next-token logits shape。",
}


def hint_for(stage: Stage, first_error: str) -> str:
    if "NotImplementedError" in first_error:
        return f"该阶段仍含学生 TODO。{HINTS[stage.key]}"
    return HINTS[stage.key]
