import json

import pytest

from jokegpt.config import ModelConfig, TrainConfig, load_configs, merge_hyperparameters


def test_head_size_divides_embedding():
    assert ModelConfig(n_embd=128, n_head=4).head_size == 32


def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=10, n_head=3)


def test_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="must be positive"):
        ModelConfig(n_layer=0)


def test_rejects_out_of_range_dropout():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.0)


def test_with_vocab_size_returns_copy_without_mutating():
    original = ModelConfig(vocab_size=34)
    updated = original.with_vocab_size(50)
    assert updated.vocab_size == 50
    assert original.vocab_size == 34


def test_config_is_frozen():
    with pytest.raises(Exception):
        ModelConfig().n_embd = 999


def test_merge_hyperparameters_includes_both_configs():
    merged = merge_hyperparameters(ModelConfig(), TrainConfig())
    assert "n_embd" in merged and "learning_rate" in merged


def test_load_configs_reads_partial_json(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"n_embd": 64, "n_head": 2, "max_iters": 5}))
    model, train = load_configs(path)
    assert (model.n_embd, model.n_head, train.max_iters) == (64, 2, 5)


def test_load_configs_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"nonsense": 1}))
    with pytest.raises(ValueError, match="unknown config keys"):
        load_configs(path)
