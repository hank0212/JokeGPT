"""Character-level tokenizer.

The notebooks built `stoi`/`itos` dicts inline from a vocab file that was kept in
Google Drive and never committed. This wraps the same logic in a class that can
save and load its vocabulary alongside a checkpoint.
"""

from __future__ import annotations

from pathlib import Path

# The character set that survives `data.cleaning.clean_text`: a newline, a space,
# the kept punctuation, and the lowercase alphabet. Sorted, it is exactly the
# 34-symbol vocabulary the shipped checkpoint was trained with.
DEFAULT_VOCAB = "".join(sorted(set("\n !',.;?" + "abcdefghijklmnopqrstuvwxyz")))


class CharTokenizer:
    """Maps single characters to integer ids and back.

    Unknown characters are dropped rather than raising, matching the original
    notebook behaviour (`[stoi[c] for c in s if c in stoi]`).
    """

    def __init__(self, vocab: str = DEFAULT_VOCAB) -> None:
        if not vocab:
            raise ValueError("vocab must not be empty")
        chars = sorted(set(vocab))
        self._chars = chars
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}

    def __len__(self) -> int:
        return len(self._chars)

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def vocab(self) -> str:
        return "".join(self._chars)

    def encode(self, text: str) -> list[int]:
        """Text -> token ids, silently skipping out-of-vocabulary characters."""
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text, silently skipping ids outside the vocabulary."""
        return "".join(self._itos[i] for i in ids if i in self._itos)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.vocab, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Derive a vocabulary from a corpus, as the notebooks did."""
        return cls("".join(sorted(set(text))))
