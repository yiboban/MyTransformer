# Assignment 06：pre-norm TransformerBlock

## 1. 学习目标与生活化直觉

Residual connection（残差连接）像保留一条高速公路：子层只学习对当前表示的修正，原信息和梯度可直接通过。Block 将“跨 token 通信”的 Attention 与“逐 token 加工”的 FFN 串联。

## 2. 固定结构与推导

$$h=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x))$$

$$y=h+\operatorname{SwiGLU}(\operatorname{RMSNorm}_2(h))$$

这是 pre-norm：norm 在子层之前。post-norm 会写成 `Norm(x+Sublayer(x))`，数值和梯度路径不同。pre-norm 的 residual 主路径更接近恒等映射，深层训练通常更稳定。

## 3. 模块位置与 shape

Block 位于 embedding 与 final norm 之间，重复 N 次。所有公开输入输出 `[B,T,D]`。例 `[2,6,32]→norm→attention→residual` 仍 `[2,6,32]`，再经 FFN/residual 仍相同；内部 QKV `[2,4,6,8]`，FFN 中间 `[2,6,64]`。

## 4. 精确 API/参数

```python
TransformerBlock(d_model, num_heads, d_ff, max_seq_len,
                 rope_theta=10000.0, device=None, dtype=None)
forward(x, position_ids=None) -> [B,T,D]
```

公开子模块名固定为 `attn_norm`, `attn`, `ffn_norm`, `ffn`，便于 fixture 严格加载。两个 RMSNorm 不能共享参数。

## 5. 实现限制与分步提示

先只构造四个子模块；forward 使用中间变量，避免一行嵌套导致误用旧 x。position ids 传给 attention。没有 dropout。可把 O projection 和 W2 置零，验证 block 精确等于输入，这同时检验两个 residual。

## 6. 常见错误

- 漏一个 residual；把第二个 residual 加回原始 x 而非 h；
- 写成 post-norm；两个 norm 复用同一对象；
- FFN 在 attention 前；
- 短序列与完整序列前缀不一致，说明 causal path 泄漏；
- batch 内样本互相影响，说明 reshape/softmax 轴错误。

## 7. 测试与失败阅读

```bash
uv run pytest tests/test_block.py -q
uv run python -m grader --stage block --verbose
```

identity test 失败先查 residual；golden 失败但 identity 通过，查 norm 顺序与子模块参数名；causal/截断失败转到 attention stage 先定位底层。

## 8. 面试题与参考答案

1. residual 为什么利于训练？提供恒等信息/梯度路径，让子层学修正。
2. pre 与 post norm？norm 分别位于子层前与 residual 后。
3. 为什么两个 norm？两个子层输入分布不同、scale 参数独立。
4. Block 是否改变 shape？不改变 D/T/B。
5. 子层输出为 0 时？两个 residual 使输出等于输入。
6. FFN 与 attention 分工？逐位置非线性加工 vs 跨位置聚合。
7. causal 性应在哪验证？attention、block、完整 LM 多层验证。
8. batch independence 是什么？同样本单跑/与其他样本拼 batch 输出一致。

## 9. 完成检查清单

- [ ] 两个独立 norm；
- [ ] 固定 pre-norm 顺序；
- [ ] 两个 residual 精确；
- [ ] golden、causal、shorten、batch invariants 通过；
- [ ] 所有参数收到有限梯度；
- [ ] block stage 全绿。
