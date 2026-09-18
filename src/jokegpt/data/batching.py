"""Turning a corpus into (input, target) batches.

Two strategies, matching the two notebooks:

  `TensorCorpus`  — encode the whole file into one tensor and slice it. Used for
                    fine-tuning, where the joke corpus fits in memory.
  `MemmapCorpus`  — memory-map the file and decode a random window per batch.
                    Used for pre-training on TinyStories/MiniPile, which do not.
"""

from __future__ import annotations

import mmap
import random
from pathlib import Path
from typing import Protocol

import torch

from ..tokenizer import CharTokenizer

TRAIN, VAL = "train", "val"


class Corpus(Protocol):
    """Anything that can hand back a training batch."""

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]: ...


class TensorCorpus:
    """Whole corpus held in one tensor, split 90/10 into train and validation."""

    def __init__(
        self,
        data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        val_fraction: float = 0.1,
    ) -> None:
        if not 0 < val_fraction < 1:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if len(data) <= block_size + 1:
            raise ValueError(
                f"corpus of {len(data)} tokens is too short for block_size {block_size}"
            )

        split_at = int((1 - val_fraction) * len(data))
        self._splits = {TRAIN: data[:split_at], VAL: data[split_at:]}
        self._block_size = block_size
        self._batch_size = batch_size
        self._device = device

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: CharTokenizer,
        block_size: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        val_fraction: float = 0.1,
    ) -> "TensorCorpus":
        text = Path(path).read_text(encoding="utf-8")
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return cls(data, block_size, batch_size, device, val_fraction)

    def __len__(self) -> int:
        return sum(len(t) for t in self._splits.values())

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._split(split)
        return _sample_windows(
            data, self._block_size, self._batch_size, self._device
        )

    def _split(self, split: str) -> torch.Tensor:
        if split not in self._splits:
            raise ValueError(f"unknown split {split!r}; expected {TRAIN!r} or {VAL!r}")
        data = self._splits[split]
        if len(data) <= self._block_size:
            raise ValueError(f"{split} split is shorter than block_size")
        return data


class MemmapCorpus:
    """Random windows read from files too large to hold in memory.

    Each call memory-maps the file, seeks to a random offset, and decodes just
    `block_size * batch_size` bytes. Because the offset can land mid-character,
    invalid UTF-8 bytes are ignored.
    """

    def __init__(
        self,
        train_path: str | Path,
        val_path: str | Path,
        tokenizer: CharTokenizer,
        block_size: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        rng: random.Random | None = None,
    ) -> None:
        self._paths = {TRAIN: Path(train_path), VAL: Path(val_path)}
        for split, path in self._paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{split} corpus not found at {path}")

        self._tokenizer = tokenizer
        self._block_size = block_size
        self._batch_size = batch_size
        self._device = device
        self._rng = rng or random.Random()

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._random_chunk(split)
        return _sample_windows(
            data, self._block_size, self._batch_size, self._device
        )

    def _random_chunk(self, split: str) -> torch.Tensor:
        if split not in self._paths:
            raise ValueError(f"unknown split {split!r}; expected {TRAIN!r} or {VAL!r}")

        span = self._block_size * self._batch_size
        with open(self._paths[split], "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if len(mm) <= span:
                    raise ValueError(
                        f"{self._paths[split]} is smaller than one batch ({span} bytes)"
                    )
                mm.seek(self._rng.randint(0, len(mm) - span))
                block = mm.read(span - 1)

        text = block.decode("utf-8", errors="ignore").replace("\r", "")
        data = torch.tensor(self._tokenizer.encode(text), dtype=torch.long)
        if len(data) <= self._block_size:
            raise ValueError("decoded chunk too short; increase batch or block size")
        return data


def _sample_windows(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` windows; targets are inputs shifted one step right."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
