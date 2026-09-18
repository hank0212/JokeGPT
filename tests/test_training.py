import pytest
import torch

from jokegpt.config import ModelConfig
from jokegpt.data.batching import TensorCorpus
from jokegpt.model import GPTLanguageModel
from jokegpt.training.checkpoint import (
    config_from_hyperparameters,
    load_checkpoint,
    save_checkpoint,
)
from jokegpt.training.logging_backends import ConsoleLogger, get_logger
from jokegpt.training.trainer import estimate_loss, train


@pytest.fixture
def corpus(tokenizer, corpus_text):
    data = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    return TensorCorpus(data, block_size=16, batch_size=4)


def test_train_returns_history_at_eval_points(tiny_model_config, tiny_train_config, corpus):
    model = GPTLanguageModel(tiny_model_config)
    history = train(model, corpus, tiny_train_config)
    assert [r.step for r in history] == [0, 2]


def test_train_updates_weights(tiny_model_config, tiny_train_config, corpus):
    model = GPTLanguageModel(tiny_model_config)
    before = model.lm_head.weight.detach().clone()
    train(model, corpus, tiny_train_config)
    assert not torch.equal(before, model.lm_head.weight)


def test_train_writes_checkpoint(tmp_path, tiny_model_config, tiny_train_config, corpus):
    path = tmp_path / "ck.pth"
    model = GPTLanguageModel(tiny_model_config)
    train(model, corpus, tiny_train_config, checkpoint_path=path)
    assert path.exists()


def test_estimate_loss_averages_only_filled_entries(tiny_model_config, corpus):
    """Regression test for the notebook bug that averaged unfilled zeros."""
    model = GPTLanguageModel(tiny_model_config)
    losses = estimate_loss(model, corpus, eval_iters=3)
    assert set(losses) == {"train", "val"}
    # An untrained model over a 34-token vocab sits near ln(34) ~= 3.5;
    # the zero-padded notebook version reported a small fraction of that.
    assert all(v > 1.0 for v in losses.values())


def test_estimate_loss_restores_training_mode(tiny_model_config, corpus):
    model = GPTLanguageModel(tiny_model_config)
    model.train()
    estimate_loss(model, corpus, eval_iters=2)
    assert model.training


def test_estimate_loss_rejects_non_positive_iters(tiny_model_config, corpus):
    model = GPTLanguageModel(tiny_model_config)
    with pytest.raises(ValueError, match="eval_iters"):
        estimate_loss(model, corpus, eval_iters=0)


def test_checkpoint_roundtrip_preserves_weights(tmp_path, tiny_model_config, tiny_train_config):
    model = GPTLanguageModel(tiny_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(model, optimizer, tiny_train_config, tmp_path / "ck.pth")

    restored = load_checkpoint(path)
    assert restored.config == tiny_model_config
    assert torch.equal(restored.model.lm_head.weight, model.lm_head.weight)


def test_load_checkpoint_restores_optimizer(tmp_path, tiny_model_config, tiny_train_config):
    model = GPTLanguageModel(tiny_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(model, optimizer, tiny_train_config, tmp_path / "ck.pth")

    fresh = GPTLanguageModel(tiny_model_config)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    load_checkpoint(path, optimizer=fresh_opt)
    assert fresh_opt.state_dict()["param_groups"][0]["lr"] == 1e-3


def test_load_checkpoint_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pth")


def test_load_checkpoint_rejects_foreign_file(tmp_path):
    path = tmp_path / "other.pth"
    torch.save({"something": 1}, path)
    with pytest.raises(ValueError, match="not a JokeGPT checkpoint"):
        load_checkpoint(path)


def test_vocab_size_inferred_from_state_dict_when_absent():
    """Legacy checkpoints omit vocab_size; recover it from lm_head."""
    hyperparameters = {"n_embd": 128, "n_head": 4, "n_layer": 4, "block_size": 128}
    state_dict = {"lm_head.weight": torch.zeros(34, 128)}
    assert config_from_hyperparameters(hyperparameters, state_dict).vocab_size == 34


def test_console_logger_is_default_backend():
    assert isinstance(get_logger("console"), ConsoleLogger)


def test_unknown_logger_rejected():
    with pytest.raises(ValueError, match="unknown logger"):
        get_logger("wandb")


def test_comet_logger_requires_credentials(monkeypatch):
    for name in ("COMET_API_KEY", "COMET_PROJECT_NAME", "COMET_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="COMET_API_KEY"):
        get_logger("comet")
