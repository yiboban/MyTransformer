# Assignment 04：Scaled Attention 与 Multi-Head Self-Attention

## 1. 学习目标与生活化直觉

Query（查询）是“我正在找什么”，Key（键）是“我能被怎样找到”，Value（值）是“找到我后取走的内容”。Attention 像按查询与目录卡的相似度分配阅读权重。

## 2. 数学与逐步推导

$$S=QK^\top/\sqrt{D_h}$$

若 q/k 各维近似零均值单位方差，未缩放点积方差约为 $D_h$；除以 $\sqrt{D_h}$ 将方差拉回常数量级，避免 softmax 饱和、梯度过小。这里必须是每个 head 的维度，不是总 `d_model`。

先 mask：

$$S'_{ij}=\begin{cases}S_{ij}&M_{ij}=\text{True}\\-\infty&\text{otherwise}\end{cases}$$

再 $P=\operatorname{softmax}(S')$、$O=PV$。若 softmax 后才乘 0，未 mask 权重不再归一化，输出幅值错误；若重新归一化还会增加不必要复杂性。

## 3. 多头不是简单重复

每个 head 读取 Q/K/V projection 中不同参数子空间，可分别学习语法、局部搭配、远距离指代等关系。若复制同一 attention 权重，多头只会重复信息。拼接各 head 后，独立 O projection 混合它们。

## 4. 模块位置与 shape

| 步骤 | shape |
|---|---|
| x | `[B,T,D]` |
| each projection | `[B,T,D]` |
| split+transpose | `[B,H,T,Dh]` |
| scores/probs | `[B,H,T,T]` |
| weighted values | `[B,H,T,Dh]` |
| merge heads | `[B,T,D]` |
| O projection | `[B,T,D]` |

例 `[2,5,32]`, H=4 → Q/K/V `[2,4,5,8]` → score `[2,4,5,5]` → output `[2,5,32]`。

## 5. 精确 API/参数

```python
scaled_dot_product_attention(q, k, v, mask=None)
MultiHeadSelfAttention(d_model, num_heads, max_seq_len,
                       rope_theta=10000.0, use_rope=True, ...)
forward(x, position_ids=None) -> [B,T,D]
```

`q_proj/k_proj/v_proj/o_proj` 都是独立无 bias `Linear`。`use_rope=False` 只用于 compatibility differential test；完整模型必须保持 True。dropout 固定为 0。

## 6. 实现限制

学生端禁用 `F.scaled_dot_product_attention`、`nn.MultiheadAttention`、所有 `nn.Transformer*`。测试端允许它们作参考。MHA PyTorch 权重映射为 `cat([Wq,Wk,Wv],0)` 和 `out_proj.weight=Wo`；其 bool mask 常用 `True=屏蔽`，所以传入前对本项目 allow mask 取反。

## 7. 分步提示

1. scaled function 先验证 4D、batch/head/Dh、K/V length；
2. `k.transpose(-1,-2)`，缩放，broadcast bool mask；
3. 处理全 mask row：本作业 causal self-attention 不产生，但通用函数应给清晰错误或定义零输出；
4. MHA 先投影，再 reshape `[B,T,H,Dh]` 与 transpose；
5. 只旋转 Q/K；创建 causal mask；
6. 合并 head 时 transpose 回去并 reshape。

## 8. 常见错误

- 除 `sqrt(d_model)`；softmax 沿 query 而不是 key；
- Q/K transpose 写反；mask bool 语义取反；
- mask 在 softmax 后乘；
- Q/K/V 共享同一个参数对象；
- 把 V 也送进 RoPE；
- batch=1 时偶然正确，batch/head>1 维错；
- 未来 token 改变过去输出。

## 9. 测试与失败阅读

```bash
uv run pytest tests/test_attention.py -q
uv run python -m grader --stage attention --verbose
```

先跑单 token、全零 Q/K、masked extreme V；再看 PyTorch differential。若 shape 对但 causal test 失败，按顺序查 mask 生成、语义、广播、应用时机。

## 10. 面试题与答案

1. Q/K/V 各自是什么？查询、可匹配地址、返回内容。
2. 为什么除 sqrt(Dh)？控制 score 方差，避免 softmax 饱和。
3. mask 为什么在 softmax 前？让禁区概率严格为 0 且其余重新归一化。
4. softmax 沿哪维？key 维，即最后一维。
5. 多头价值？独立 projection 学不同关系子空间。
6. 为什么有 O projection？混合各 head 并映回模型空间。
7. causal invariant？改变未来输入不能改变过去输出。
8. 复杂度？标准 self-attention 时间/score 内存约 $O(T^2)$。

## 11. 完成检查清单

- [ ] 手算均匀/单 token/mask；
- [ ] PyTorch SDPA 与 MHA differential 通过；
- [ ] QKV/O 独立且 RoPE 只进 QK；
- [ ] causal、batch independence、gradient 通过；
- [ ] attention stage 全绿。
