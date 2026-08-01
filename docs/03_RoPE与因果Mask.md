# Assignment 03：RoPE 与因果 Mask

## 1. 学习目标与直觉

RoPE（Rotary Positional Embedding，旋转位置编码）像给每个位置的二维指针转不同角度；比较两个指针时，点积自然只关心角度差。因果 mask 像遮住答案纸右侧：位置 t 只能看自己和过去。

## 2. 从二维旋转推导

二维向量旋转位置 $p$ 的角度 $p\theta$：

$$
R(p\theta)=\begin{bmatrix}\cos p\theta&-\sin p\theta\\
\sin p\theta&\cos p\theta\end{bmatrix}
$$

高维向量按 `(0,1),(2,3),...` 分成独立二维平面，各平面频率：

$$
\omega_i=\Theta^{-2i/D_h}
$$

两位置的 Q/K 点积：

$$
(R(p)q)^\top(R(s)k)=q^\top R(p)^\top R(s)k=q^\top R(s-p)k
$$

因此只出现相对位置 $s-p$。Q 和 K 都旋转才能得到上述差值；V 是被加权汇总的内容，不参与“地址相似度”，旋转 V 会把位置坐标混入内容并破坏语义。

## 3. 模块位置、shape 与 API

`x[B,T,D] → Q/K/V[B,H,T,Dh] → RoPE(Q,K) → scores`。

```python
RotaryPositionalEmbedding(head_dim, max_seq_len, theta=10000.0, ...)
forward(x: [B,H,T,Dh], position_ids: [T] or [B,T] | None) -> same
make_causal_mask(query_length, key_length=None, device=None) -> [Q,K] bool
```

例：`D=32,H=4,Dh=8,T=6`，RoPE 输入输出 `[2,4,6,8]`；mask `[6,6]` 广播到 `[2,4,6,6]`。

## 4. mask 精确定义

本项目 `True=允许`、`False=屏蔽`。方阵为下三角。矩形 `[Q,K]` 将 Q 解释为 K 序列末尾的 Q 个 query：

$$M_{ij}=[j\le i+(K-Q)]$$

要求 `Q>0,K>0,Q<=K`。mask 在 score softmax 前使用。

## 5. 参数含义与边界

`head_dim` 必须为偶数；`max_seq_len` 是最大 position + 1；`theta` 控制频率尺度。显式 position id 必须是整数、非负且小于最大长度。position 0 的 cos=1/sin=0，所以输出严格保持。

## 6. 分步实现提示

1. 构造偶数维频率向量；
2. position 与频率做外积得到角度；
3. 用 `x[...,0::2]` 与 `x[...,1::2]` 分组；
4. 计算 `even*cos-odd*sin`、`even*sin+odd*cos` 后交错还原；
5. 让 cos/sin 转到 x dtype/device；
6. mask 用 `arange` 比较，别创建浮点矩阵。

## 7. 实现限制与常见错误

- 不对 V 旋转；不把 D_model 当 head_dim；
- 奇数维必须报错，不静默丢最后一维；
- position `[B,T]` 广播时要插入 head 轴；
- 将前半/后半维配对不是本项目约定；
- mask 语义与 PyTorch 某些 API 相反，比较前要确认/取反；
- max length 越界要报清晰异常。

## 8. 测试与失败阅读

```bash
uv run pytest tests/test_rope.py tests/test_mask.py -q
uv run python -m grader --stage rope --verbose
```

先检查 position0 identity 和 pair norm invariant；若这两项通过但 golden 失败，多半是配对约定、theta 指数或 position 广播不同。

## 9. 面试题与参考答案

1. 为什么两两维度？二维是连续旋转的最小实数表示。
2. 相对位置从哪来？$R(p)^TR(s)=R(s-p)$。
3. 为什么 Q/K 都转？点积才能消去绝对角、留下差。
4. 为什么 V 不转？V 是内容载体，不决定注意力地址。
5. position0？恒等旋转。
6. theta 影响？不同维度频率与可表达距离尺度。
7. causal mask 的 True？本项目表示允许。
8. 为何 mask 可 `[T,T]`？前导 batch/head 维可广播。

## 10. 完成检查清单

- [ ] 二维公式能手推；
- [ ] position0、pair norm、explicit positions 通过；
- [ ] 奇数维/越界错误清晰；
- [ ] 3×3 与矩形 mask 精确；
- [ ] rope stage 全绿。
