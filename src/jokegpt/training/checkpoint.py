"""Saving and restoring training state.

The checkpoint format is the notebooks' own — a dict of `model_state_dict`,
`optimizer_state_dict` and `hyperparameters` — so `checkpoints/model_128_4_4.pth`
still loads. What is new is that loading *derives* the model config from the
stored hyperparameters rather than trusting whatever globals happen to be set,
which is what made the notebooks silently load a checkpoint into a mismatched
architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import ModelConfig, TrainConfig, merge_hyperparameters
from ..model import GPTLanguageModel


@dataclass(frozen=True)
class Checkpoint:
    """A loaded checkpoint: the model, its config, and the raw hyperparameters."""

    model: GPTLanguageModel
    config: ModelConfig
    hyperparameters: dict


def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    path: str | Path,
) -> Path:
    """Write model weights, optimizer state and hyperparameters to `path`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparameters": merge_hyperparameters(model.config, train_config),
        },
        target,
    )
    return target


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    override: ModelConfig | None = None,
) -> Checkpoint:
    """Rebuild a model from `path`.

    The architecture comes from the checkpoint's own hyperparameters unless
    `override` is given. Pass `optimizer` to also restore optimizer state, which
    is what makes fine-tuning resume rather than restart.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No checkpoint at {source}")

    raw = torch.load(source, map_location=device, weights_only=False)
    for key in ("model_state_dict", "hyperparameters"):
        if key not in raw:
            raise ValueError(f"{source} is missing {key!r}; not a JokeGPT checkpoint")

    hyperparameters = raw["hyperparameters"]
    config = override or config_from_hyperparameters(
        hyperparameters, raw["model_state_dict"]
    )

    model = GPTLanguageModel(config)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device)

    if optimizer is not None:
        if "optimizer_state_dict" not in raw:
            raise ValueError(f"{source} has no optimizer state to restore")
        optimizer.load_state_dict(raw["optimizer_state_dict"])

    return Checkpoint(model=model, config=config, hyperparameters=hyperparameters)


def config_from_hyperparameters(
    hyperparameters: dict, state_dict: dict | None = None
) -> ModelConfig:
    """Reconstruct a `ModelConfig` from a checkpoint's stored hyperparameters.

    Older checkpoints (including the shipped one) do not record `vocab_size`, so
    it is recovered from the shape of the output layer when a state dict is given.
    """
    fields = {
        name: hyperparameters[name]
        for name in ModelConfig.__dataclass_fields__
        if name in hyperparameters
    }

    if "vocab_size" not in fields and state_dict is not None:
        if "lm_head.weight" not in state_dict:
            raise ValueError("cannot infer vocab_size: no lm_head.weight in state dict")
        fields["vocab_size"] = state_dict["lm_head.weight"].shape[0]

    return ModelConfig(**fields)
