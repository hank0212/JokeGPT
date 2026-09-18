#!/usr/bin/env python3
"""Fine-tune a pre-trained checkpoint on the joke corpus.

Loads the base model's architecture from its checkpoint, so the two cannot drift
apart the way they could in the notebooks.

Usage:
    python scripts/finetune.py --base checkpoints/model_128_4_4.pth \
        --corpus data/jokes.txt --out checkpoints/jokegpt.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from jokegpt.config import TrainConfig
from jokegpt.data.batching import TensorCorpus
from jokegpt.tokenizer import CharTokenizer
from jokegpt.training.checkpoint import load_checkpoint
from jokegpt.training.logging_backends import get_logger
from jokegpt.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Pre-trained checkpoint")
    parser.add_argument("--corpus", type=Path, required=True, help="Cleaned joke corpus")
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/jokegpt.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also restore the base checkpoint's optimizer state",
    )
    parser.add_argument("--logger", choices=["console", "comet"], default="console")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = CharTokenizer.load(args.vocab) if args.vocab else CharTokenizer()

    checkpoint = load_checkpoint(args.base, device=device)
    model = checkpoint.model
    print(f"loaded base model: {checkpoint.config}")

    if model.config.vocab_size != tokenizer.vocab_size:
        raise SystemExit(
            f"Vocab mismatch: checkpoint expects {model.config.vocab_size} tokens, "
            f"tokenizer has {tokenizer.vocab_size}. Fine-tuning must reuse the "
            "pre-training vocabulary."
        )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    if args.resume_optimizer:
        load_checkpoint(args.base, device=device, optimizer=optimizer)

    corpus = TensorCorpus.from_file(
        path=args.corpus,
        tokenizer=tokenizer,
        block_size=model.config.block_size,
        batch_size=train_config.batch_size,
        device=device,
    )
    print(f"corpus: {len(corpus):,} tokens | device: {device}")

    history = train(
        model=model,
        corpus=corpus,
        config=train_config,
        optimizer=optimizer,
        logger=get_logger(args.logger),
        checkpoint_path=args.out,
    )

    final = history[-1]
    print(f"final: train {final.train_loss:.4f} | val {final.val_loss:.4f}")
    print(f"checkpoint: {args.out}")


if __name__ == "__main__":
    main()
