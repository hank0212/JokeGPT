import pytest
import torch

from jokegpt.model import GPTLanguageModel


def test_forward_returns_logits_shaped_by_vocab(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    idx = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, tiny_model_config.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    idx = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    _, loss = model(idx, idx)
    assert loss.ndim == 0 and loss.item() > 0


def test_rejects_sequence_longer_than_block_size(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    too_long = torch.zeros((1, tiny_model_config.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(too_long)


def test_generate_appends_requested_token_count(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)
    assert out.shape == (1, 6)


def test_generate_crops_context_beyond_block_size(tiny_model_config):
    """Generation must keep working once the sequence outgrows the window."""
    model = GPTLanguageModel(tiny_model_config)
    start = torch.zeros((1, tiny_model_config.block_size), dtype=torch.long)
    out = model.generate(start, max_new_tokens=3)
    assert out.shape == (1, tiny_model_config.block_size + 3)


def test_generate_restores_training_mode(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    model.train()
    model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=2)
    assert model.training


def test_generate_rejects_non_positive_temperature(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.zeros((1, 1), dtype=torch.long), 2, temperature=0)


def test_top_k_restricts_sampled_tokens(tiny_model_config):
    """With top_k=1 sampling is deterministic given fixed weights."""
    model = GPTLanguageModel(tiny_model_config)
    start = torch.zeros((1, 1), dtype=torch.long)
    first = model.generate(start, max_new_tokens=8, top_k=1)
    second = model.generate(start, max_new_tokens=8, top_k=1)
    assert torch.equal(first, second)


def test_causal_mask_blocks_future_tokens(tiny_model_config):
    """Changing a later token must not change an earlier position's logits."""
    model = GPTLanguageModel(tiny_model_config)
    model.eval()
    a = torch.randint(0, tiny_model_config.vocab_size, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % tiny_model_config.vocab_size
    with torch.no_grad():
        logits_a, _ = model(a)
        logits_b, _ = model(b)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)


def test_num_parameters_counts_all_weights(tiny_model_config):
    model = GPTLanguageModel(tiny_model_config)
    assert model.num_parameters == sum(p.numel() for p in model.parameters())
