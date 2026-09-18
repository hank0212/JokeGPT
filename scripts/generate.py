#!/usr/bin/env python3
"""Sample text from a trained checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/model_128_4_4.pth \
        --prompt "why did the chicken" --max-new-tokens 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from jokegpt.generate import generate_text
from jokegpt.tokenizer import CharTokenizer
from jokegpt.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/model_128_4_4.pth")
    )
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--prompt", default="", help="Empty prompt samples unconditionally")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = CharTokenizer.load(args.vocab) if args.vocab else CharTokenizer()
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    if checkpoint.config.vocab_size != tokenizer.vocab_size:
        raise SystemExit(
            f"Vocab mismatch: checkpoint expects {checkpoint.config.vocab_size} "
            f"tokens, tokenizer has {tokenizer.vocab_size}."
        )

    for i in range(args.samples):
        text = generate_text(
            model=checkpoint.model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        if args.samples > 1:
            print(f"--- sample {i + 1} ---")
        print(text)


if __name__ == "__main__":
    main()
