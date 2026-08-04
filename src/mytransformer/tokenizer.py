"""Assignment 01: UTF-8 byte-level BPE 学生骨架。"""
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PRETOKEN_PATTERN = regex.compile(
    r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"
)

# 隔离special token special token 跳过BPE训练
def split_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    """将普通文本和 special token 分开。"""
    if not special_tokens:
        return [text]
    # 长的 special token 放在前面。
    # 例如 <A><B> 必须比 <A> 更早匹配。
    ordered_special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )
    escaped_tokens = [
        regex.escape(token)
        for token in ordered_special_tokens
    ]
    special_pattern = (
        "(" + "|".join(escaped_tokens) + ")"
    )

    return regex.split(special_pattern, text)

# token转byte
def token_to_byte(text:str)->tuple[bytes,...]:
    encoded = text.encode("utf-8")
    # print(encoded)
    return tuple(bytes([byte_value]) for byte_value in encoded)

# 实现不重叠合并 输入序列 pair
def merge_pair(symbols:tuple[bytes,...],pair:tuple[bytes,bytes]) -> tuple[bytes,...]:
    result = []
    i = 0
    n = len(symbols)
    while i < n:
        if i+1 < n and symbols[i]==pair[0] and symbols[i+1]==pair[1]:
            result.append(symbols[i]+symbols[i+1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)

def train_bpe(
    input_path: str | Path, # 数据集txt路径
    vocab_size: int, # vocab大小
    special_tokens: list[str], # 特殊token
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 byte-level BPE；完整规则见 docs/01_BPE.md。"""
    # TODO(A01): 读取 UTF-8 语料、pre-tokenize、确定性统计 pair 并依次 merge。
    # print(input_path)
    # print(vocab_size)
    # print(special_tokens)
    temp = Path(input_path)
    content = temp.read_text(encoding="utf-8")
    # print("==========================================")
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(
            "vocab_size太小"
        )
    # 初始化byte词表
    vocab = {i : bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = special_token.encode("utf-8")
    # 记录每次merge 加入到vocab
    merges: list[tuple[bytes,bytes]] = []
    # print(vocab)
    segments = split_special_tokens(content,special_tokens)
    # 转换成集合
    special_token_set = set(special_tokens)
    # 计数器 统计每种 pre-token byte出现了多少次
    pretoken_frequencies = Counter()

    # print(special_token_set)
    # 隔离special tokens
    for segment in segments:
        if segment == '':
            continue
        if segment in special_token_set:
            continue
            # print("special:",repr(segment))
        else:
            parts = PRETOKEN_PATTERN.findall(segment)
            # 转成 UTF-8 byte symbols
            # BPE不能直接拿字符串训练 需要把每个 pre-token 转成一个个单独的byte
            for part in parts:
                byte_symbols = token_to_byte(part) # (b'h', b'e', b'l', b'l', b'o')
                # 统计每种 pre-token byte出现了多少次
                pretoken_frequencies[byte_symbols] += 1
                #print("normal:",repr(part))
                #print("bytes:",token_to_byte(part))
    # print(pretoken_frequencies)
    # 统计相邻byte pair
    # 不断执行BPE merge 直到词表达到要求大小 或无法合并
    # print(len(vocab),vocab_size)
    while len(vocab) < vocab_size:
        # 计数器 统计每个pair的总出现次数
        pair_counts = Counter()
        for symbols, frequency in pretoken_frequencies.items():
            #print("当前序列:",symbols)
            #print("出现频次:",frequency)
            for pair in zip(symbols,symbols[1:], strict=False):
                pair_counts[pair] += frequency
        #print(pair_counts)
        # 找出首先应该合并的pair 如果没有可以合并的pair pair_counts就是空的
        if not pair_counts:
            break
        best_pair = min(pair_counts,key=lambda temp:(-pair_counts[temp],temp))
        merges.append(best_pair)
        new_piece = best_pair[0] + best_pair[1]
        # 加到vocab中
        new_token_id = len(vocab)
        vocab[new_token_id] = new_piece
        #print("本轮 merge：", best_pair)
        #print("新 token：", new_piece)
        #print("新 token id：", new_token_id)
        #print("当前 merges：", merges)
        new_pretoken_frequencies = Counter()
        for symbols, frequency in pretoken_frequencies.items():
            merged_symbols = merge_pair(symbols,best_pair)
            new_pretoken_frequencies[merged_symbols] += frequency
        #print("合并前：", pretoken_frequencies)
        #print("合并后：", new_pretoken_frequencies)
        pretoken_frequencies = new_pretoken_frequencies
    #print(len(vocab))
    #print(merges)
    return vocab,merges




class BPETokenizer:
    """严格按 merge rank 编码、支持不可拆分 special token 的 tokenizer。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # TODO(A01): 建立 byte->id、merge rank 和 longest-match special token 索引。
        # 保存词表 用于解码
        self.vocab = vocab
        # 建立反向词表: bytes->token id 用于编码
        self.byte_to_id = {a:b for b,a in vocab.items()}
        # 建立merge pair -> rank 编码的时候 rank越小越优先
        self.merge_ranks = {pair:rank for rank,pair in enumerate(merges)}
        # None转成空列表
        self.special_tokens = special_tokens if special_tokens is not None else []
        # special token字符串 -> token id
        self.special_to_id: dict[str,int] = {}
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8") # 必须转成bytes才能和词表一致
            if special_bytes not in self.byte_to_id:
                raise ValueError("vocab缺少special token")
            self.special_to_id[special_token] = self.byte_to_id[special_bytes]
        # 最长special token优先匹配
        self.ordered_special_tokens = sorted(self.special_tokens,key=len,reverse=True)

    def encode(self, text: str) -> list[int]:
        tokens_id:list[int] = []
        segments = split_special_tokens(text,self.special_tokens)
        for segment in segments:
            if segment == "":
                continue
            if segment in self.special_to_id:
                tokens_id.append(self.special_to_id[segment])
            else:
                normal_ids = self._encode_normal_text(segment)
                tokens_id.extend(normal_ids)
        return tokens_id
    # 边处理边产生
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # 找到对应的bytes
        byte_pieces:list[bytes] = []
        for id in ids:
            if id not in self.vocab:
                raise ValueError(f"Unknown token id: {id}")
            byte_pieces.append(self.vocab[id])
        # print(byte_pieces)
        result = b"".join(byte_pieces)
        return result.decode("utf-8",errors="replace")

    # 编码一个普通pre-token
    def _encode_pretoken(self,text:str)->list[int]:
        symbols = token_to_byte(text)
        while True:
            candidate_pairs = []
            for pair in zip(symbols,symbols[1:], strict=False):
                if pair in self.merge_ranks:
                    candidate_pairs.append(pair)
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs,key=lambda a:self.merge_ranks[a])
            symbols = merge_pair(symbols,best_pair)
        return [self.byte_to_id[symbol] for symbol in symbols]

    # 编码一整段不含special token的普通文本
    def _encode_normal_text(self,text:str)->list[int]:
        tokens_id:list[int] = []
        parts = PRETOKEN_PATTERN.findall(text)
        for part in parts:
            parts_id = self._encode_pretoken(part)
            tokens_id.extend(parts_id)
        return tokens_id
