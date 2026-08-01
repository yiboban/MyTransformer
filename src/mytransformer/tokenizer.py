"""Assignment 01: UTF-8 byte-level BPE 学生骨架。"""

from collections.abc import Iterable, Iterator
from pathlib import Path


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 byte-level BPE；完整规则见 docs/01_BPE.md。"""
    # TODO(A01): 读取 UTF-8 语料、pre-tokenize、确定性统计 pair 并依次 merge。
    raise NotImplementedError("Assignment 01: 请实现 train_bpe；参见 docs/01_BPE.md")


class BPETokenizer:
    """严格按 merge rank 编码、支持不可拆分 special token 的 tokenizer。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # TODO(A01): 建立 byte->id、merge rank 和 longest-match special token 索引。
        raise NotImplementedError("Assignment 01: 请实现 BPETokenizer.__init__")

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Assignment 01: 请实现 BPETokenizer.encode")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError("Assignment 01: 请实现 BPETokenizer.encode_iterable")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("Assignment 01: 请实现 BPETokenizer.decode")
