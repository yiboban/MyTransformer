"""Assignment 02: 数值函数学生骨架。"""

from torch import Tensor


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """用减最大值技巧实现数值稳定 softmax。"""
    # TODO(A02): 不得调用 torch.softmax/F.softmax。
    raise NotImplementedError("Assignment 02: 请实现 stable_softmax")


def next_token_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """返回所有 token 位置的平均 next-token cross entropy。"""
    # TODO(A02): 验证 logits[..., V] 与 targets[...] shape，不得调用 F.cross_entropy。
    raise NotImplementedError("Assignment 02: 请实现 next_token_cross_entropy")
