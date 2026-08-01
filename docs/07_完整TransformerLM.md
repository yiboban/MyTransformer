# Assignment 07：完整 Transformer Language Model

## 1. 学习目标与生活化直觉

把此前零件装成一台“下一个 token 预测器”。每个位置输出整个词表的 logits；训练像给每一位置一张下一字符答案，生成则把模型刚选出的 token 接回输入继续预测。

## 2. 固定结构与数学

```text
ids → token_embedding → N×Block → final RMSNorm → independent lm_head → logits
```

给序列 $t_0,...,t_{T-1}$，输入前 T-1 个 token，标签右移：

$$x=(t_0,...,t_{T-2}),\qquad y=(t_1,...,t_{T-1})$$

训练最小化 $-\sum_i\log p(t_{i+1}\mid t_{\le i})$。logits 不是 probability，不要求非负或和为 1；loss 或 sampling 内部才 softmax。

## 3. config/API 与参数

```python
TransformerConfig(vocab_size, context_length, d_model,
                  num_layers, num_heads, d_ff, rope_theta=10000.0)
TransformerLM(config, device=None, dtype=None)
forward(token_ids: [B,T]) -> [B,T,V]
generate(token_ids, max_new_tokens, temperature=1.0) -> [B,T+new]
```

config 所有整数必须正；D 可被 H 整除；Dh 为偶数。T 不超过 context。`temperature=0` 定义 greedy；正温度用 `logits/temperature` 采样。生成超过 context 时只把最后 `context_length` token 喂给模型，但返回完整历史。

## 4. shape 流转例

默认小配置 V=271, context=16, D=32, L=2, H=4, Dff=64：

| 步骤 | shape |
|---|---|
| ids | `[2,10]` |
| embedding | `[2,10,32]` |
| each block | `[2,10,32]` |
| norm | `[2,10,32]` |
| lm_head | `[2,10,271]` |

CPU 即可测试，不需要下载大数据集；8GB 4060 远超所需。

## 5. state_dict 与训练 step

lm_head 默认不与 embedding weight tying（权重绑定），便于明确检查两套参数。保存 `model.state_dict()`，相同 config 新模型 strict load 后应输出一致。一次标准 step：zero_grad → forward → shifted CE → backward → AdamW.step；所有应训练参数 gradient 非空、有限，至少一个参数变化。

## 6. 实现限制与分步提示

- 不用 `nn.Transformer*`，不使用 dropout；
- `nn.ModuleList` 注册 N 个独立 block，不能用同一对象重复引用；
- forward 验证 2D 整数 ids、范围、context；
- generate 每轮取最后位置 logits；greedy 用 argmax，采样可用 multinomial；
- 保持 batch，返回 long ids；验证 `max_new_tokens>=0` 与 temperature；
- 不在测试里硬编码输出，golden 来自固定 state_dict。

## 7. 集成 invariant

1. golden logits；2. future token 不影响过去；3. 单跑与 batch 一致；4. state_dict 往返；5. backward 梯度有限；6. AdamW 更新；7. finite loss；8. optional tiny overfit；9. generation 长度正确；10. 同 seed 初始化/输出一致。

## 8. 常见错误

- 用当前位置 token 当 label，未右移；
- 对 logits 先 argmax 再算 loss；
- lm_head 与 embedding 意外共享对象；
- ModuleList 里重复同一个 block；
- generation 永远输入完整历史，超过 context 报错；
- 温度 0 做除法产生 Inf/NaN；
- causal mask 缺失导致训练 loss 看似更低但未来泄漏。

## 9. 测试与失败阅读

```bash
uv run python -m grader --stage lm
uv run pytest tests/test_transformer_lm.py -m "lm and not slow" -q
uv run pytest tests/test_transformer_lm.py -m slow -q
```

先保证下层六个 stage 通过。golden mismatch 从第一层参数/输出逐层二分；future leak 先回归 attention；loss NaN 检查 softmax、mask 与学习率；state roundtrip 检查 persistent buffer/参数名。

## 10. 面试题与参考答案

1. Decoder-only 含义？只有带因果 self-attention 的解码器堆栈，无 encoder/cross-attention。
2. logits 为什么不是概率？线性头输出未归一化实数。
3. 标签如何右移？输入去尾、目标去首。
4. 为什么 final norm？控制堆栈输出尺度再映射词表。
5. weight tying 的利弊？省参数/共享语义，但参数耦合；本作业默认关闭。
6. greedy 与 sampling？argmax 确定；按分布采样有多样性。
7. temperature？缩放 logits；低温更尖，高温更平。
8. 为什么 causal invariant 重要？否则训练偷看答案，推理不可用。
9. state_dict 不含什么？代码、config 类和优化器状态（除非另存）。
10. tiny overfit 用途？快速验证整个训练数据流可学习，不代表泛化。

## 11. 完成检查清单

- [ ] config 边界检查；
- [ ] embedding/blocks/norm/独立 head 顺序正确；
- [ ] 9 类集成测试通过；
- [ ] generation greedy 与 context crop 正确；
- [ ] optional slow loss 明显下降；
- [ ] `grader --all` 100/100。
