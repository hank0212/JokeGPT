"""Build the fine-tuning joke corpus from the three source datasets.

The notebook cloned these repos into Colab and concatenated them inline. Here
each source is a named loader that returns a `pandas.Series` of raw joke text,
so a missing dataset degrades to a warning instead of a NameError halfway down.

Sources (clone these next to each other, see README):
  - taivop/joke-dataset          reddit_jokes.json, stupidstuff.json, wocka.json
  - shuttie/dadjokes             train.csv, test.csv
  - amoudgl/short-jokes-dataset  shortjokes.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cleaning import clean_text, is_valid_line


def load_joke_dataset(root: str | Path) -> pd.Series:
    """taivop/joke-dataset: reddit (title + body), stupidstuff, wocka."""
    root = Path(root)
    reddit = pd.read_json(_require(root / "reddit_jokes.json"))
    stupid = pd.read_json(_require(root / "stupidstuff.json"))
    wocka = pd.read_json(_require(root / "wocka.json"))

    return pd.concat(
        [
            reddit["title"].fillna("") + " " + reddit["body"].fillna(""),
            stupid["body"],
            wocka["body"],
        ],
        ignore_index=True,
    )


def load_dadjokes(root: str | Path) -> pd.Series:
    """shuttie/dadjokes: question + response, train and test splits."""
    root = Path(root)
    frames = [
        pd.read_csv(_require(root / name))
        for name in ("train.csv", "test.csv")
    ]
    return pd.concat(
        [df["question"].fillna("") + " " + df["response"].fillna("") for df in frames],
        ignore_index=True,
    )


def load_short_jokes(root: str | Path) -> pd.Series:
    """amoudgl/short-jokes-dataset: a single `Joke` column."""
    return pd.read_csv(_require(Path(root) / "shortjokes.csv"))["Joke"]


def build_corpus(
    sources: list[pd.Series], trim_sentences: bool = True
) -> list[str]:
    """Concatenate, de-duplicate, clean and filter every joke.

    De-duplication happens before cleaning (as in the notebook) and the valid
    line filter after, since cleaning can empty a line out entirely.

    `trim_sentences=False` disables the setup-destroying truncation described in
    `cleaning._trim_to_sentence_bounds`.
    """
    if not sources:
        raise ValueError("no joke sources supplied")

    combined = pd.concat(sources, axis=0, ignore_index=True).drop_duplicates()
    cleaned = combined.apply(clean_text, trim_sentences=trim_sentences)
    return [line for line in cleaned if is_valid_line(line)]


def write_corpus(lines: list[str], path: str | Path) -> Path:
    """Write one cleaned joke per line, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset file at {path}. See the README for the clone commands."
        )
    return path
