"""Assignment 02: 数值函数学生骨架。"""
import torch
from torch import Tensor


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """用减最大值技巧实现数值稳定 softmax。"""
    max_values = x.max(dim=dim,keepdim=True).values
    shifted = x - max_values
    xx = torch.exp(shifted)
    s = torch.sum(xx,dim=dim,keepdim=True)
    # print(xx/s)
    return xx / s

def next_token_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """返回所有 token 位置的平均 next-token cross entropy。"""
    if logits.ndim < 1:
        raise ValueError("logits must have one dim at least")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("shape is not match")
    # print(logits.shape)
    # print(targets.shape)
    m = logits.max(dim=-1,keepdim=True).values
    shifted = logits - m
    logsumexp = m + torch.log(torch.sum(torch.exp(shifted),dim=-1,keepdim=True))
    # zy怎么拿
    selected = torch.gather(logits,dim=-1,index=targets.unsqueeze(-1))
    loss = (logsumexp - selected).squeeze(-1)
    loss_mean = loss.mean()
    return loss_mean
