"""Model and training configuration.

The original notebooks kept every hyperparameter as a module-level global, which
meant the model classes could only be instantiated inside that one notebook.
These frozen dataclasses make the same values explicit and injectable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the character-level GPT.

    Defaults match the shipped `checkpoints/model_128_4_4.pth`.
    """

    vocab_size: int = 34
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    def with_vocab_size(self, vocab_size: int) -> "ModelConfig":
        """Return a copy bound to a tokenizer's vocabulary (never mutates self)."""
        return replace(self, vocab_size=vocab_size)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings for one training run."""

    batch_size: int = 64
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 200
    learning_rate: float = 3e-4
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_iters <= 0:
            raise ValueError("batch_size and max_iters must be positive")
        if self.eval_interval <= 0 or self.eval_iters <= 0:
            raise ValueError("eval_interval and eval_iters must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


def merge_hyperparameters(
    model: ModelConfig, train: TrainConfig
) -> dict[str, int | float]:
    """Flatten both configs into the single dict the checkpoints store."""
    return {**asdict(model), **asdict(train)}


def load_configs(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    """Read a JSON file holding any subset of the config fields."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(raw).__name__}")

    model_fields = {f for f in ModelConfig.__dataclass_fields__}
    train_fields = {f for f in TrainConfig.__dataclass_fields__}
    unknown = set(raw) - model_fields - train_fields
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")

    return (
        ModelConfig(**{k: v for k, v in raw.items() if k in model_fields}),
        TrainConfig(**{k: v for k, v in raw.items() if k in train_fields}),
    )
