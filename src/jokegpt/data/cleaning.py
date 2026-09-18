"""Text normalisation shared by pre-training and fine-tuning.

The notebooks redefined `clean_text` four times with small divergences. This is
the consolidated version: the fine-tuning variant, which is the one that
produced the shipped checkpoint's 34-character vocabulary.

Each step is a compiled pattern so the function can be mapped over millions of
lines without re-compiling regexes per call.
"""

from __future__ import annotations

import re

# Ordered (pattern, replacement) pairs applied in sequence.
_WORDS_WITH_DIGITS = re.compile(r"\w*\d\w*")
_LONG_DIGIT_RUNS = re.compile(r"\d{5,}")
_VERY_LONG_WORDS = re.compile(r"\b\w{21,}\b")
_URLS = re.compile(r"http\S+")
_EMAILS = re.compile(r"\S+@\S+")
_DISALLOWED_CHARS = re.compile(r"[^a-z0-9.,!?;\s']")
_INTRAWORD_PUNCT = re.compile(r"\s\w+[.,!?;]\w+\s")
_REPEATED_COMMAS = re.compile(r"\,+[,\s]+[^\w]")
_REPEATED_PERIODS = re.compile(r"\.+[\.\s]+[^\w]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?])")
_COLLAPSE_PERIODS = re.compile(r"[.]+")
_COLLAPSE_COMMAS = re.compile(r"[,]+")
_PERIOD_THEN_COMMA = re.compile(r"\.+[,]+")
_WHITESPACE = re.compile(r"\s+")
_HAS_LETTER = re.compile(r"[a-z]")

MIN_VALID_LINE_LENGTH = 2


def clean_text(text: object, trim_sentences: bool = True) -> str:
    """Normalise one line of raw corpus text to the model's character set.

    Lowercases, strips anything outside `[a-z0-9.,!?;\\s']`, removes URLs,
    emails and numeric tokens, then tidies up the punctuation the earlier steps
    leave behind.

    `trim_sentences` controls the notebook's first-period/last-period truncation.
    It defaults to True to reproduce the shipped checkpoint's training data, but
    see `_trim_to_sentence_bounds` for why you probably want it off.
    """
    text = str(text).lower()

    text = _URLS.sub("", text)
    text = _EMAILS.sub("", text)
    text = _WORDS_WITH_DIGITS.sub("", text)
    text = _LONG_DIGIT_RUNS.sub("", text)
    text = _VERY_LONG_WORDS.sub("", text)
    text = _DISALLOWED_CHARS.sub("", text)
    text = _INTRAWORD_PUNCT.sub(" ", text)
    text = " ".join(text.split())

    if trim_sentences:
        text = _trim_to_sentence_bounds(text)

    text = _REPEATED_COMMAS.sub(", ", text)
    text = _REPEATED_PERIODS.sub(". ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _COLLAPSE_PERIODS.sub(".", text)
    text = _COLLAPSE_COMMAS.sub(",", text)
    text = _PERIOD_THEN_COMMA.sub(",", text)

    return text


def _trim_to_sentence_bounds(text: str) -> str:
    """Keep only the span between the first and the last period.

    The intent in the original notebooks was to drop partial sentences at the
    start and end of scraped text. On jokes it misfires badly: any joke whose
    setup ends in a period loses the setup, keeping only the punchline.

        "i told my wife she was drawing her eyebrows too high. she looked
         surprised."  ->  "she looked surprised"

    Jokes whose setup ends in "?" are unaffected, which is why dad jokes survive
    and one-liners do not. Kept as the default for checkpoint fidelity; pass
    `trim_sentences=False` to `clean_text` to disable it.
    """
    start = text.find(".") + 1
    end = text.rfind(".")
    if start != 0 and end != -1 and end > start:
        return text[start:end].strip()
    return text


def is_valid_line(line: str) -> bool:
    """Whether a cleaned line is worth keeping in the corpus."""
    if not line:
        return False
    if len(line) < MIN_VALID_LINE_LENGTH:
        return False
    return bool(_HAS_LETTER.search(line))
