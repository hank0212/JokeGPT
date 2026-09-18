"""Sampling text from a trained model."""

from __future__ import annotations

import torch

from .model import GPTLanguageModel
from .tokenizer import CharTokenizer


def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | str = "cpu",
) -> str:
    """Continue `prompt` for `max_new_tokens` characters.

    An empty prompt starts from token 0, matching the notebooks' unconditional
    `torch.zeros((1, 1))` seed. The prompt is included in the returned string.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    ids = tokenizer.encode(prompt)
    context = (
        torch.tensor([ids], dtype=torch.long, device=device)
        if ids
        else torch.zeros((1, 1), dtype=torch.long, device=device)
    )

    out = model.generate(
        context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    return tokenizer.decode(out[0].tolist())
