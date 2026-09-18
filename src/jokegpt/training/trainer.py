"""The training loop, shared by pre-training and fine-tuning.

Both notebooks had their own near-identical copy of this loop. The only real
difference was which corpus fed it, so `train` takes the corpus as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import TrainConfig, merge_hyperparameters
from ..data.batching import TRAIN, VAL, Corpus
from ..model import GPTLanguageModel
from .checkpoint import save_checkpoint
from .logging_backends import ConsoleLogger, MetricLogger


@dataclass(frozen=True)
class EvalResult:
    """Mean loss on each split at one point in training."""

    step: int
    train_loss: float
    val_loss: float


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel, corpus: Corpus, eval_iters: int
) -> dict[str, float]:
    """Average the loss over `eval_iters` batches from each split.

    Note: the fine-tuning notebook allocated its `losses` buffer once outside the
    split loop but only filled the first 100 entries, then averaged the whole
    buffer — so every reported loss was scaled down by the unfilled zeros. This
    version allocates per split and fills exactly what it averages.
    """
    if eval_iters <= 0:
        raise ValueError(f"eval_iters must be positive, got {eval_iters}")

    was_training = model.training
    model.eval()
    try:
        out = {}
        for split in (TRAIN, VAL):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = corpus.get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    finally:
        model.train(was_training)
    return out


def train(
    model: GPTLanguageModel,
    corpus: Corpus,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
    logger: MetricLogger | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[EvalResult]:
    """Run `config.max_iters` optimisation steps, returning the eval history.

    A checkpoint is written at every evaluation point when `checkpoint_path` is
    set, so a crashed long run does not lose everything.
    """
    logger = logger or ConsoleLogger()
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )
    torch.manual_seed(config.seed)

    logger.log_parameters(merge_hyperparameters(model.config, config))
    print(f"{model.num_parameters / 1e6:.2f}M parameters")

    history: list[EvalResult] = []
    model.train()

    for step in range(config.max_iters):
        x, y = corpus.get_batch(TRAIN)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        is_last = step == config.max_iters - 1
        if step % config.eval_interval == 0 or is_last:
            losses = estimate_loss(model, corpus, config.eval_iters)
            result = EvalResult(step, losses[TRAIN], losses[VAL])
            history.append(result)

            logger.log_metric("train_loss", result.train_loss, step)
            logger.log_metric("val_loss", result.val_loss, step)

            if checkpoint_path is not None:
                save_checkpoint(model, optimizer, config, checkpoint_path)

    logger.end()
    return history
