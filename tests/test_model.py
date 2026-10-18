from __future__ import annotations

import math

import pytest
import torch

from secondchair.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY).eval()


def test_logits_shape_and_initial_loss(model: GPT) -> None:
    idx = torch.randint(TINY.vocab_size, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, TINY.vocab_size)
    assert loss is not None and abs(loss.item() - math.log(TINY.vocab_size)) < 0.5  # ~uniform at init


def test_no_targets_no_loss(model: GPT) -> None:
    assert model(torch.zeros(1, 4, dtype=torch.long))[1] is None


def test_attention_is_causal(model: GPT) -> None:
    idx = torch.randint(TINY.vocab_size, (1, 16))
    changed = idx.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % TINY.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)  # earlier positions never see later tokens
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_rejects_sequences_longer_than_block_size(model: GPT) -> None:
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_input_and_output_embeddings_are_tied(model: GPT) -> None:
    assert model.head.weight is model.tok_emb.weight


def test_default_size_is_a_few_million_parameters() -> None:
    assert 2_000_000 < GPT(GPTConfig(vocab_size=229)).num_params() < 5_000_000


def test_learns_a_repeating_pattern() -> None:
    torch.manual_seed(0)
    model = GPT(TINY)
    pattern = torch.arange(8).repeat(2)[None, :16]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(pattern[:, :-1], pattern[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1
