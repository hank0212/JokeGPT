#!/usr/bin/env python3
"""Fetch the joke datasets (and optionally the pre-training corpus).

Clones the three joke sources into --data-root, skipping any already present.

Usage:
    python scripts/download_data.py                  # joke datasets (~100 MB)
    python scripts/download_data.py --tinystories    # also TinyStories (~2 GB)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

JOKE_REPOS = {
    "joke-dataset": "https://github.com/taivop/joke-dataset.git",
    "short-jokes-dataset": "https://github.com/amoudgl/short-jokes-dataset.git",
    "dadjokes": "https://huggingface.co/datasets/shuttie/dadjokes",
}

TINYSTORIES = {
    "TinyStories-train.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "TinyStories-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}


def clone(name: str, url: str, root: Path) -> None:
    target = root / name
    if target.exists():
        print(f"  {name}: already present, skipping")
        return

    print(f"  {name}: cloning from {url}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  {name}: FAILED\n{result.stderr.strip()}", file=sys.stderr)


def download(name: str, url: str, root: Path) -> None:
    target = root / name
    if target.exists():
        print(f"  {name}: already present, skipping")
        return

    print(f"  {name}: downloading (this is large)")
    try:
        with urllib.request.urlopen(url) as response, open(target, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:  # noqa: BLE001 - report and continue
        target.unlink(missing_ok=True)
        print(f"  {name}: FAILED ({exc})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--tinystories",
        action="store_true",
        help="Also download the TinyStories pre-training corpus (~2 GB)",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        raise SystemExit("git is required but was not found on PATH")

    args.data_root.mkdir(parents=True, exist_ok=True)

    print(f"joke datasets -> {args.data_root}")
    for name, url in JOKE_REPOS.items():
        clone(name, url, args.data_root)

    if args.tinystories:
        print(f"\ntinystories -> {args.data_root}")
        for name, url in TINYSTORIES.items():
            download(name, url, args.data_root)

    print("\nDone. Next: python scripts/prepare_jokes.py --data-root", args.data_root)


if __name__ == "__main__":
    main()
