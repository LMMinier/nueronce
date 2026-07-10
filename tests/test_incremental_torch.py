"""Torch-gated equivalence contract for nueronce.incremental.IncrementalGenerator.

Run wherever torch is installed (desktop/Colab): greedy stepwise logits must
match the dense forward, and greedy generation must be byte-identical to a
manual dense argmax loop over the same context rule."""

import pytest

torch = pytest.importorskip("torch")

from nueronce.incremental import IncrementalGenerator
from nueronce.model import NUERONCEModel, ModelConfig


def _small_model(seed=0):
    torch.manual_seed(seed)
    m = NUERONCEModel(ModelConfig(
        byte_embed_dim=8, d_local=12, d_model=16, p_max=8, physical_blocks=1,
        logical_depth=2, n_heads=2, unit_window=6, decoder_window=8,
        decoder_layers=2, d_state=4, channel_dim=4, min_patch=2, max_patch=6))
    m.eval()
    return m


@torch.no_grad()
def _dense_greedy(model, prompt: bytes, max_new: int, max_ctx: int) -> bytes:
    ids = list(prompt) or [32]
    for _ in range(max_new):
        ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long)
        logits, _ = model(ctx)
        ids.append(int(logits[0, -1].argmax()))
    return bytes(ids)


def test_stepwise_logits_match_dense_forward():
    m = _small_model()
    inc = IncrementalGenerator(m)
    inc.prime(list(b"Boundary cases: punctuation, and words!"))
    for extra in b" More bytes arrive one at a time.":
        fast = inc._last_logits()
        dense_logits, _ = m(torch.tensor([inc.ids], dtype=torch.long))
        dense = dense_logits[0, -1]
        assert torch.allclose(fast, dense, atol=1e-5), (fast - dense).abs().max()
        assert int(fast.argmax()) == int(dense.argmax())
        inc._append(int(extra))


@pytest.mark.parametrize("prompt", [b"Hello world, this is a test.",
                                    b"def add(a, b):\n    return",
                                    b"word " * 12])
def test_greedy_generation_byte_identical(prompt):
    m = _small_model(1)
    inc = IncrementalGenerator(m)
    dense = _dense_greedy(m, prompt, max_new=24, max_ctx=96)
    fast = inc.generate(prompt, max_new=24, greedy=True, max_ctx=96,
                        continuation_only=False)
    assert fast == dense


def test_greedy_identical_through_window_slide():
    m = _small_model(2)
    inc = IncrementalGenerator(m)
    prompt = b"The quick brown fox jumps over the lazy dog. " * 2
    dense = _dense_greedy(m, prompt, max_new=30, max_ctx=40)
    fast = inc.generate(prompt, max_new=30, greedy=True, max_ctx=40,
                        continuation_only=False)
    assert fast == dense
