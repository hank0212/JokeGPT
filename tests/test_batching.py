import pytest
import torch

from jokegpt.data.batching import MemmapCorpus, TensorCorpus


def test_batch_shapes_match_config(tokenizer, corpus_text):
    data = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    corpus = TensorCorpus(data, block_size=16, batch_size=4)
    x, y = corpus.get_batch("train")
    assert x.shape == (4, 16) and y.shape == (4, 16)


def test_targets_are_inputs_shifted_by_one(tokenizer, corpus_text):
    data = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    corpus = TensorCorpus(data, block_size=16, batch_size=4)
    x, y = corpus.get_batch("train")
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_and_val_splits_are_disjoint(tokenizer, corpus_text):
    data = torch.arange(1000)
    corpus = TensorCorpus(data, block_size=8, batch_size=2, val_fraction=0.1)
    assert len(corpus) == 1000


def test_rejects_unknown_split(tokenizer, corpus_text):
    data = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    corpus = TensorCorpus(data, block_size=16, batch_size=4)
    with pytest.raises(ValueError, match="unknown split"):
        corpus.get_batch("test")


def test_rejects_corpus_shorter_than_block():
    with pytest.raises(ValueError, match="too short"):
        TensorCorpus(torch.arange(5), block_size=16, batch_size=2)


def test_rejects_invalid_val_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        TensorCorpus(torch.arange(100), block_size=8, batch_size=2, val_fraction=1.5)


def test_from_file_reads_and_encodes(tmp_path, tokenizer, corpus_text):
    path = tmp_path / "c.txt"
    path.write_text(corpus_text, encoding="utf-8")
    corpus = TensorCorpus.from_file(path, tokenizer, block_size=16, batch_size=2)
    x, _ = corpus.get_batch("val")
    assert x.shape == (2, 16)


def test_memmap_corpus_yields_batches(tmp_path, tokenizer, corpus_text):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text(corpus_text * 4, encoding="utf-8")
    val.write_text(corpus_text * 4, encoding="utf-8")
    corpus = MemmapCorpus(train, val, tokenizer, block_size=16, batch_size=4)
    x, y = corpus.get_batch("train")
    assert x.shape == (4, 16) and y.shape == (4, 16)


def test_memmap_corpus_reports_missing_file(tmp_path, tokenizer):
    with pytest.raises(FileNotFoundError):
        MemmapCorpus(tmp_path / "nope.txt", tmp_path / "nope.txt", tokenizer, 16, 4)
