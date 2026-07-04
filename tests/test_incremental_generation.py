"""The incremental generator must EARN its wiring: byte-identical greedy
output and float-tolerance logits vs the dense path, including the
window-sliding regime. No torch needed."""

import numpy as np

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from cfna.microtorch.incremental import IncrementalGenerator
from cfna.microtorch.tensor import no_grad


def _small_model(seed=0):
    np.random.seed(seed)
    return MicroCFNAModel(MicroModelConfig(
        byte_embed_dim=8, d_local=12, d_model=16, p_max=8, physical_blocks=1,
        logical_depth=2, n_heads=2, unit_window=6, decoder_window=8,
        decoder_layers=2, d_state=4, channel_dim=4, min_patch=2, max_patch=6))


PROMPTS = [
    b"Hello world, this is a test.",
    b"def add(a, b):\n    return",
    b"a",
    b"word " * 12,
]


def test_greedy_output_byte_identical():
    m = _small_model()
    inc = IncrementalGenerator(m)
    for prompt in PROMPTS:
        dense = m.generate(prompt, max_new=24, greedy=True, max_ctx=96)
        fast = inc.generate(prompt, max_new=24, greedy=True, max_ctx=96)
        assert fast == dense, (prompt, fast, dense)


def test_greedy_identical_through_window_slide():
    # max_ctx smaller than prompt+new forces the dense path to slide; the
    # incremental path must re-prime and stay byte-identical.
    m = _small_model(1)
    inc = IncrementalGenerator(m)
    prompt = b"The quick brown fox jumps over the lazy dog. " * 2
    dense = m.generate(prompt, max_new=30, greedy=True, max_ctx=40)
    fast = inc.generate(prompt, max_new=30, greedy=True, max_ctx=40)
    assert fast == dense


def test_stepwise_logits_match_dense_forward():
    m = _small_model(2)
    inc = IncrementalGenerator(m)
    ids = list(b"Boundary cases: punctuation, and words!")
    with no_grad():
        inc.prime(ids)
        for extra in b" More bytes arrive one at a time.":
            fast = inc._last_logits()
            dense_logits, _ = m.forward(np.array([inc.ids]))
            dense = dense_logits.data[0, -1]
            assert np.allclose(fast, dense, atol=1e-9), np.abs(fast - dense).max()
            assert int(fast.argmax()) == int(dense.argmax())
            inc._append(int(extra))


def test_stop_bytes_and_min_new_contract():
    m = _small_model(3)
    inc = IncrementalGenerator(m)
    dense = m.generate(b"User: hi", max_new=40, greedy=True, max_ctx=96,
                       stop_bytes=b"\n", min_new=4)
    fast = inc.generate(b"User: hi", max_new=40, greedy=True, max_ctx=96,
                        stop_bytes=b"\n", min_new=4)
    assert fast == dense
