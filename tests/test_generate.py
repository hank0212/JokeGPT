import pytest

from jokegpt.generate import generate_text
from jokegpt.model import GPTLanguageModel


def test_generated_text_includes_prompt(tiny_model_config, tokenizer):
    model = GPTLanguageModel(tiny_model_config)
    out = generate_text(model, tokenizer, prompt="why", max_new_tokens=10)
    assert out.startswith("why") and len(out) > 3


def test_empty_prompt_still_generates(tiny_model_config, tokenizer):
    model = GPTLanguageModel(tiny_model_config)
    out = generate_text(model, tokenizer, prompt="", max_new_tokens=10)
    assert len(out) > 0


def test_all_output_characters_are_in_vocab(tiny_model_config, tokenizer):
    model = GPTLanguageModel(tiny_model_config)
    out = generate_text(model, tokenizer, prompt="a", max_new_tokens=30)
    assert set(out) <= set(tokenizer.vocab)


def test_rejects_non_positive_token_count(tiny_model_config, tokenizer):
    model = GPTLanguageModel(tiny_model_config)
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_text(model, tokenizer, max_new_tokens=0)
