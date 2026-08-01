# Assignment 01：Byte-level BPE

## 1. 学习目标与生活化直觉

Byte Pair Encoding（字节对编码）像给常见字节片段建立“快捷短语”：所有原始 UTF-8 字节永远可表示，再把频繁相邻片段逐步合并，以更少 token 表示常见文本。

## 2. 数学与训练推导

先用 `regex` 的 GPT-2 风格规则切 pre-token，避免 merge 跨越不合理边界。每个 pre-token 转 UTF-8 bytes，并初始拆为单字节 symbol。第 r 轮统计相邻 pair：

$$
c(a,b)=\sum_w f(w)\,\#_w(a,b)
$$

选 `c` 最大者；并列时选 `(a,b)` 按 bytes 元组字典序最小者。将该 pair 的所有不重叠出现替换成 `a+b`，追加 merge rank。初始普通 vocab 是 `0..255`；special token 按传入顺序从 256 开始；随后才是 merge 新 token。若语料已无 pair，可早停，因此最终 vocab 允许小于请求大小。

推荐 pre-token pattern：GPT-2 家族的 contraction、letter、number、punctuation、whitespace 分支；必须用第三方 `regex`，因为普通 `re` 不支持 `\p{L}`/`\p{N}`。

## 3. 在完整模型中的位置

文本先由 tokenizer 变成 `[B,T]` id，LM 只处理 id。模型输出 id 后再 decode 成 UTF-8 文本。Tokenizer 没有梯度，不属于 `nn.Module`。

## 4. Shape/API/参数

```python
train_bpe(input_path, vocab_size: int, special_tokens: list[str])
BPETokenizer(vocab, merges, special_tokens=None)
encode(text: str) -> list[int]
encode_iterable(iterable) -> Iterator[int]
decode(ids: list[int]) -> str
```

| 对象 | 类型/shape | 含义 |
|---|---|---|
| vocab | `dict[int, bytes]` | id 到 byte piece |
| merges | `list[(bytes,bytes)]` | 索引就是 merge rank |
| text | Python str | Unicode 文本 |
| ids | `list[int]` | 1D token 流 |

例：“abc” UTF-8 为 `[97,98,99]`；若 rank0=`a+b→ab`、rank1=`ab+c→abc`，则编码从 `[a,b,c]→[ab,c]→[abc]`。

## 5. Special token 规则

Special token 不参加普通 pre-tokenization/merge。扫描输入时若多个 special 在同一位置匹配，选择字节/字符长度最长者；相同长度按传入顺序。连续 special 要分别成为原子 token。例如 `<A><B>` 同时是 special 时，它优先于 `<A>` 后接普通文本。

## 6. 编码为什么必须按 rank

编码不是“从 vocab 里找最长字符串”。每一步在当前 symbol 序列中找所有可 merge pair 中 rank 最小者，再合并其不重叠出现，直到没有候选。训练历史定义了 tokenization 语义。

## 7. 实现限制与分步提示

- 禁止 tiktoken、HF tokenizers、sentencepiece 或其他成品 BPE；
- 用 bytes，不用 Unicode 字符作为训练 symbol；
- 先实现 `_merge_pair(symbols,pair)` 小助手并手测重叠；
- 统计结果的 tie-break 必须显式写进 key，不能依赖 dict 遍历；
- decode 先拼 bytes，再 `decode("utf-8", errors="replace")`；本测试合法 roundtrip 不产生 replacement；
- 非法 id 抛出含 id 的清晰 `ValueError` 或 `KeyError`。

## 8. 常见错误

- 将“你”当 1 个初始 symbol，实际上 UTF-8 是 3 bytes；
- special token 先被普通 regex 拆分；
- 频次并列使用 `max(dict)` 或插入顺序，导致平台不稳定；
- 对每个单词贪心最长匹配，不遵守 rank；
- `encode_iterable` 返回 list 而非 lazy iterator；
- vocab_size 小于 `256 + special_count` 未报错。

## 9. 测试与失败阅读

```bash
uv run pytest tests/test_bpe_train.py -q
uv run pytest tests/test_tokenizer.py -q
uv run python -m grader --stage bpe --verbose
```

训练与编解码测试分开。若 known first merge 错，先打印 pair count 和 tie key；若 roundtrip 错，打印 `text.encode()`、piece bytes、拼接 bytes，不要只看渲染后的字符。

## 10. 面试题与参考答案

1. 为什么 byte-level 无 OOV？任意 UTF-8 最终由 0..255 构成。
2. BPE 的 rank 是什么？merge 在训练列表中的先后优先级。
3. 为什么 pre-tokenize？控制 merge 边界并减少统计规模。
4. 为什么普通 `re` 不够？需要 Unicode property class。
5. 并列为何显式 tie-break？保证可复现，不受遍历顺序影响。
6. special token 为什么原子化？它承载控制语义，拆开会丢失边界。
7. longest match 解决什么？重叠 special 的歧义。
8. encode/decode roundtrip 的前提？vocab pieces 保留原 UTF-8 bytes，special 映射一致。
9. BPE 的缺点？形态边界不总自然，tokenization 依赖语料与 merge budget。

## 11. 完成检查清单

- [ ] 0..255 精确映射；
- [ ] special 顺序正确且不可拆；
- [ ] GPT-2 风格 `regex` pre-tokenization；
- [ ] pair count、重叠 merge、tie-break 确定；
- [ ] 编码严格按 rank；
- [ ] 中英文、emoji、空串、连续 special roundtrip；
- [ ] BPE stage 全绿。
