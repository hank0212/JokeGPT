#!/usr/bin/env python3
"""Build the cleaned joke corpus from the three source datasets.

Usage:
    python scripts/prepare_jokes.py --data-root data/raw --out data/jokes.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path side effect)

from jokegpt.data.jokes import (
    build_corpus,
    load_dadjokes,
    load_joke_dataset,
    load_short_jokes,
    write_corpus,
)
from jokegpt.tokenizer import DEFAULT_VOCAB, CharTokenizer

LOADERS = {
    "joke-dataset": load_joke_dataset,
    "dadjokes": load_dadjokes,
    "short-jokes-dataset": load_short_jokes,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the cloned dataset repos",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/jokes.txt"), help="Output corpus file"
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("data/vocab.txt"),
        help="Where to write the vocabulary derived from the corpus",
    )
    parser.add_argument(
        "--no-trim-sentences",
        action="store_true",
        help=(
            "Disable the first-period/last-period truncation. The original "
            "notebooks had it on, which strips the setup from any joke whose "
            "setup ends in a period. Recommended for new training runs."
        ),
    )
    args = parser.parse_args()

    sources = []
    for name, loader in LOADERS.items():
        directory = args.data_root / name
        if not directory.exists():
            print(f"skipping {name}: {directory} not found")
            continue
        series = loader(directory)
        print(f"loaded {name}: {len(series):,} rows")
        sources.append(series)

    if not sources:
        raise SystemExit(
            f"No datasets found under {args.data_root}. See the README for clone commands."
        )

    lines = build_corpus(sources, trim_sentences=not args.no_trim_sentences)
    path = write_corpus(lines, args.out)
    print(f"wrote {len(lines):,} cleaned jokes to {path}")

    tokenizer = CharTokenizer.from_text("\n".join(lines))
    tokenizer.save(args.vocab_out)
    print(f"wrote {tokenizer.vocab_size}-character vocab to {args.vocab_out}")
    print(f"vocab: {tokenizer.vocab!r}")

    if tokenizer.vocab != DEFAULT_VOCAB:
        print(
            "\nNote: this vocabulary differs from the one the shipped checkpoint "
            "was trained with. Do NOT pass it to finetune.py against "
            "checkpoints/model_128_4_4.pth -- fine-tuning must reuse the "
            "pre-training vocabulary. Omit --vocab to use the built-in default."
        )


if __name__ == "__main__":
    main()
