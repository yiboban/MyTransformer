from __future__ import annotations

import pytest

from grader.adapters import BPETokenizer


def tiny_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"abc"
    vocab[258] = b"<A>"
    vocab[259] = b"<A><B>"
    return vocab, [(b"a", b"b"), (b"ab", b"c")]


@pytest.mark.bpe
@pytest.mark.parametrize("text", ["", "hello", "中文🙂", "a\n b", "é"])
def test_tokenizer_roundtrip_unicode_and_empty(text):
    vocab, merges = tiny_vocab()
    tokenizer = BPETokenizer(vocab, merges)
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.mark.bpe
def test_tokenizer_obeys_merge_rank_not_greedy_longest_piece():
    vocab, merges = tiny_vocab()
    tokenizer = BPETokenizer(vocab, merges)
    assert tokenizer.encode("abc") == [257]


@pytest.mark.bpe
def test_tokenizer_special_tokens_use_longest_match_and_stay_atomic():
    vocab, merges = tiny_vocab()
    tokenizer = BPETokenizer(vocab, merges, ["<A>", "<A><B>"])
    ids = tokenizer.encode("<A><B><A>")
    assert ids == [259, 258]
    assert tokenizer.decode(ids) == "<A><B><A>"


@pytest.mark.bpe
def test_tokenizer_encode_iterable_is_lazy_flat_stream():
    vocab, merges = tiny_vocab()
    tokenizer = BPETokenizer(vocab, merges)
    stream = tokenizer.encode_iterable(iter(["ab", "c"]))
    assert iter(stream) is stream
    assert list(stream) == [256, ord("c")]


@pytest.mark.bpe
def test_tokenizer_decode_rejects_unknown_id_with_clear_message():
    vocab, merges = tiny_vocab()
    tokenizer = BPETokenizer(vocab, merges)
    with pytest.raises((KeyError, ValueError), match="999|token|id"):
        tokenizer.decode([999])
