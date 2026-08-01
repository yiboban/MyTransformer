from __future__ import annotations

import pytest

from grader.adapters import train_bpe


@pytest.mark.bpe
def test_train_bpe_starts_with_all_bytes_and_specials_in_order(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("aba aba", encoding="utf-8")
    vocab, merges = train_bpe(corpus, 259, ["<eos>", "<pad>"])
    assert [vocab[i] for i in range(256)] == [bytes([i]) for i in range(256)]
    assert [vocab[256], vocab[257]] == [b"<eos>", b"<pad>"]
    assert len(vocab) == 259
    assert len(merges) == 1


@pytest.mark.bpe
def test_train_bpe_known_first_merge_and_is_repeatable(tmp_path):
    corpus = tmp_path / "known.txt"
    corpus.write_text("abab ab zz", encoding="utf-8")
    result1 = train_bpe(corpus, 257, [])
    result2 = train_bpe(corpus, 257, [])
    assert result1 == result2
    assert result1[1] == [(b"a", b"b")]


@pytest.mark.bpe
def test_train_bpe_tie_breaks_by_lexicographically_smallest_byte_pair(tmp_path):
    corpus = tmp_path / "tie.txt"
    corpus.write_text("ab\ncd", encoding="utf-8")
    _, merges = train_bpe(corpus, 257, [])
    assert merges == [(b"a", b"b")]


@pytest.mark.bpe
def test_train_bpe_counts_utf8_bytes_not_unicode_codepoints(tmp_path):
    corpus = tmp_path / "unicode.txt"
    corpus.write_text("你你", encoding="utf-8")
    vocab, _ = train_bpe(corpus, 257, [])
    assert all(isinstance(piece, bytes) for piece in vocab.values())


@pytest.mark.bpe
def test_train_bpe_rejects_impossible_vocab_size(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        train_bpe(corpus, 255, [])
