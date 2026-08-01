# Assignment 05：SiLU 与 SwiGLU FFN

## 1. 学习目标与直觉

Attention 在 token 之间“搬运信息”，FFN 在每个 token 内部“加工信息”。SwiGLU 像一条内容支路乘上一扇连续可调的门：两条上投影各司其职，再由下投影回到模型维度。

## 2. 数学推导与模块位置

$$\operatorname{SiLU}(x)=x\sigma(x),\qquad \sigma(x)=\frac1{1+e^{-x}}$$

$$\operatorname{SwiGLU}(x)=W_2(\operatorname{SiLU}(W_1x)\odot W_3x)$$

三个 Linear 的原因：W1 产生经非线性的 gate/content 表示，W3 提供另一条线性内容/门支路，W2 将 $D_{ff}$ 压回 D。若漏 W3，就退化为普通两层 SiLU FFN。

## 3. shape/API/参数

```python
silu(x) -> same
SwiGLU(d_model, d_ff, device=None, dtype=None)
forward(x: [B,T,D]) -> [B,T,D]
```

| 步骤 | shape |
|---|---|
| x | `[B,T,D]` |
| W1x, W3x | `[B,T,Dff]` |
| elementwise product | `[B,T,Dff]` |
| W2 | `[B,T,D]` |

例 `[3,7,32]→两条[3,7,64]→乘积[3,7,64]→[3,7,32]`。

## 4. 实现限制与分步提示

三个 Linear 均无 bias，`d_ff` 显式传入，不在内部“聪明地”重算。先实现 `silu=x*sigmoid(x)`，然后建立命名 `w1/w2/w3`；forward 的括号必须对应公式。不要调用高层 SwiGLU；测试用 `F.silu` 仅作 oracle。

## 5. 常见错误

- 写成 `W2(silu(W1x * W3x))`，非线性位置不同；
- 漏掉 W3 或把加法当乘法；
- W2 方向写成 D→Dff；
- 三层共享 weight；
- 只对 `[B,T,D]` 中某个固定 B/T 有效；
- 参数或输入 gradient 为 None/NaN。

## 6. 测试与失败阅读

```bash
uv run pytest tests/test_ffn.py -q
uv run python -m grader --stage ffn --verbose
```

若 SiLU differential 通过而模块不通过，分别打印 W1/W3 输出、逐元素乘积和 W2 输出 shape；若参数列表少于三项，检查是否复用了对象。

## 7. 面试题与参考答案

1. SiLU 与 ReLU 差异？SiLU 平滑、负区间不完全为零。
2. SwiGLU 为什么三层 Linear？两条上投影形成 gate×content，第三层下投影。
3. 元素乘要求？两支 shape 相同 `[B,T,Dff]`。
4. FFN 混合 token 吗？不混合，每个位置共享同一函数。
5. Dff 作用？控制每 token 的非线性容量与计算量。
6. 为什么无 bias？本项目统一现代简化架构与清晰测试布局。
7. 漏 W3 会怎样？退化、参数/表达能力与 fixture 都不同。

## 8. 完成检查清单

- [ ] SiLU 公式、梯度正确；
- [ ] w1/w2/w3 参数独立且无 bias；
- [ ] 任意 B/T shape 正确；
- [ ] 所有参数有限梯度；
- [ ] ffn stage 全绿。
