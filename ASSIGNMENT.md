# 总作业说明

## 目标与最终结构

你将实现以下数据流：

```text
UTF-8 文本 → byte-level BPE token ids
token ids [B,T] → Embedding → N × pre-norm TransformerBlock
→ final RMSNorm → 独立 lm_head → logits [B,T,V]
```

每个 block 固定为：

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Attention 中 Q/K 使用 RoPE，V 不使用；bool mask 里 `True=允许关注`。没有 dropout，确保小模型测试确定且可复现。

## 完成顺序与公开 API

| Assignment | 文件 | 公开 API | 分值 | 命令 |
|---|---|---|---:|---|
| 00 | docs/00 | 张量与测试约定 | 0 | 阅读 |
| 01 | tokenizer.py | `train_bpe`, `BPETokenizer` | 20 | `python -m grader --stage bpe` |
| 02 | layers.py, functional.py | `Linear`, `Embedding`, `RMSNorm`, softmax, CE | 10 | `--stage primitives` |
| 03 | rope.py | `RotaryPositionalEmbedding`, `make_causal_mask` | 15 | `--stage rope` |
| 04 | attention.py | scaled attention, MHA | 20 | `--stage attention` |
| 05 | ffn.py | `silu`, `SwiGLU` | 10 | `--stage ffn` |
| 06 | block.py | `TransformerBlock` | 10 | `--stage block` |
| 07 | model.py | config, LM, generation | 15 | `--stage lm` |

所有 tensor API 默认 `batch_first=True`。完整 shape 见 A00；精确构造器签名以学生骨架和分模块文档为准。

## 全局实现规则

- 固定随机种子；不得依赖 GPU、网络、绝对路径或测试顺序。
- 参数、输入与输出正确传播 dtype/device；不在 forward 中偷偷搬到 CPU。
- 浮点使用 `torch.testing.assert_close`；token id、merge rank、bool mask 精确比较。
- 不硬编码 fixture 输出；正确实现应支持合理的不同 batch/sequence/head。
- 输入不合法时给 `ValueError`/`TypeError`，消息指出哪一维或哪个参数有问题。
- 禁用 API 由 AST 检查，不是脆弱的字符串搜索。

## Assignment 01：Byte-level BPE

初始普通词表严格为 `0..255 → bytes([id])`，special token 随后按传入顺序加入。使用 `regex` 包执行 GPT-2 风格 pre-tokenization；训练时每轮选全语料最高频 pair，频次相同时选 `(left_bytes, right_bytes)` 按 Python bytes 元组字典序最小者。编码必须重复应用全局最低 merge rank，而不是在词表中贪心找最长普通 token。special token 使用 longest match 并保持原子性。详见 `docs/01_BPE.md`。

## Assignment 02：基础模块

手写无 bias Linear、Embedding、最后一维 RMSNorm、减最大值 softmax 和 token 平均 cross entropy。禁止调用骨架注释列出的同功能高层 API。权重布局跟 `F.linear` 一致，便于显式 differential test。

## Assignment 03～04：位置与注意力

RoPE 将最后一维按 `(0,1),(2,3),...` 成对旋转；position 0 是恒等映射。mask 必须先作用于 score，再 softmax。scaled attention 的缩放项是 `sqrt(d_head)`。MHA 必须有独立 Q/K/V/O 权重，可用 `use_rope=False` 与 PyTorch MHA 比较，但完整模型必须启用 RoPE。

## Assignment 05～07：FFN、Block 与 LM

SwiGLU 使用三个无 bias Linear：

$$
W_2(\operatorname{SiLU}(W_1x)\odot W_3x)
$$

Block 是严格 pre-norm 和两个 residual。LM 使用 token embedding、block stack、final RMSNorm、独立 lm_head。训练标签右移一位：用 `ids[:, :-1]` 预测 `ids[:, 1:]`。`generate(..., temperature=0)` 定义为 greedy argmax；正温度可采样。

## 测试 oracle 与 golden fixture

测试不把某个 PyTorch API 当作唯一答案，而是组合：手算、独立公式、PyTorch differential、梯度、invariant 和四个 tiny golden fixture。fixture 含版本、dtype、config、输入、固定权重（RoPE 无参数除外）、期望输出；相邻 JSON manifest 保存文件 SHA-256，加载优先使用 `weights_only=True`。

## 评分器与调试流程

先运行一个 stage；摘要给出首个关键错误、可能原因、讲义和精确 pytest 命令。需要原始 traceback 时加 `--verbose`。推荐循环：

1. 读当前讲义和 API；
2. 写最小 TODO；
3. 跑该 stage；
4. 从第一个失败开始，打印 shape 而不是猜；
5. stage 全绿后再进入下一项；
6. 最后跑 `--all` 和 optional slow test。

## 完成标准

- `ruff check .` 通过；
- `pytest tests -m "not slow"` 通过；
- `python -m grader --all` 得 100；
- 可选 `pytest tests -m slow` 明显降 loss；
- 没有删除测试、降低 tolerance、在 adapter 实现算法或调用禁用 API。
