import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jokegpt.config import ModelConfig, TrainConfig  # noqa: E402
from jokegpt.tokenizer import CharTokenizer  # noqa: E402


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def tiny_model_config(tokenizer) -> ModelConfig:
    """A model small enough to train for a few steps on CPU in a test."""
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        n_embd=16,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )


@pytest.fixture
def tiny_train_config() -> TrainConfig:
    return TrainConfig(batch_size=4, max_iters=3, eval_interval=2, eval_iters=2)


@pytest.fixture
def corpus_text() -> str:
    return ("why did the chicken cross the road? to get to the other side. " * 200)
