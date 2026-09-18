import pytest

from jokegpt.tokenizer import DEFAULT_VOCAB, CharTokenizer


def test_default_vocab_matches_shipped_checkpoint(tokenizer):
    # checkpoints/model_128_4_4.pth has a 34-row lm_head.
    assert tokenizer.vocab_size == 34


def test_roundtrip_preserves_in_vocab_text(tokenizer):
    text = "why did the chicken cross the road?"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_out_of_vocab_characters_are_dropped(tokenizer):
    assert tokenizer.decode(tokenizer.encode("HELLO@#$world")) == "world"


def test_decode_ignores_out_of_range_ids(tokenizer):
    assert tokenizer.decode([0, 9999, 1]) == tokenizer.decode([0, 1])


def test_from_text_derives_sorted_unique_vocab():
    assert CharTokenizer.from_text("cba cba").vocab == " abc"


def test_empty_vocab_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        CharTokenizer("")


def test_save_and_load_roundtrip(tmp_path, tokenizer):
    path = tmp_path / "vocab.txt"
    tokenizer.save(path)
    assert CharTokenizer.load(path).vocab == tokenizer.vocab


def test_default_vocab_is_sorted_and_unique():
    assert DEFAULT_VOCAB == "".join(sorted(set(DEFAULT_VOCAB)))
