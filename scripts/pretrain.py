#!/usr/bin/env python3
"""Pre-train the base model on a large prose corpus (TinyStories / MiniPile).

Reads batches by memory-mapping the corpus, so the files need not fit in RAM.

Usage:
    python scripts/pretrain.py --train-file data/TinyStories-train.txt \
        --val-file data/TinyStories-valid.txt --out checkpoints/base.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from jokegpt.config import ModelConfig, TrainConfig
from jokegpt.data.batching import MemmapCorpus
from jokegpt.model import GPTLanguageModel
from jokegpt.tokenizer import CharTokenizer
from jokegpt.training.logging_backends import get_logger
from jokegpt.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=None, help="Vocab file (optional)")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/base.pth"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--logger", choices=["console", "comet"], default="console")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = CharTokenizer.load(args.vocab) if args.vocab else CharTokenizer()

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
    )

    corpus = MemmapCorpus(
        train_path=args.train_file,
        val_path=args.val_file,
        tokenizer=tokenizer,
        block_size=model_config.block_size,
        batch_size=train_config.batch_size,
        device=device,
    )

    model = GPTLanguageModel(model_config).to(device)
    print(f"device: {device}")

    history = train(
        model=model,
        corpus=corpus,
        config=train_config,
        logger=get_logger(args.logger),
        checkpoint_path=args.out,
    )

    final = history[-1]
    print(f"final: train {final.train_loss:.4f} | val {final.val_loss:.4f}")
    print(f"checkpoint: {args.out}")


if __name__ == "__main__":
    main()
