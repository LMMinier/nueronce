"""Parity gate for the C++ engine (cpp/cfna_engine.cpp) against the microtorch
oracle. The engine ships only if these pass:

- dense last-position logits match the NumPy float64 forward to <= 1e-8
  on a random small model AND on the real trained 112K checkpoint;
- greedy generation (C++ incremental path) is byte-identical to the oracle's
  dense greedy generate, including the window-sliding regime.

Skipped cleanly when no C++ toolchain is available.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from cfna.microtorch.optim import AdamW
from cfna.training.sharded_sft import save_checkpoint

REPO = Path(__file__).resolve().parents[1]
REAL_CKPT = REPO / "checkpoints" / "micro_cfna_sft_100k" / "best.pt"

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="no g++")


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("cpp") / "cfna_run"
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(out),
                    str(REPO / "cpp" / "cfna_engine.cpp")], check=True)
    return out


def _export(model, path):
    sys.path.insert(0, str(REPO / "scripts"))
    from export_cfna_bin import export
    export(model, str(path))


def _small_model(seed=7):
    np.random.seed(seed)
    return MicroCFNAModel(MicroModelConfig(
        byte_embed_dim=8, d_local=12, d_model=16, p_max=8, physical_blocks=2,
        logical_depth=3, n_heads=2, unit_window=6, decoder_window=8,
        decoder_layers=2, d_state=4, channel_dim=4, min_patch=2, max_patch=6))


def _cpp_logits(binary, bin_path, prompt: bytes, max_ctx=96):
    r = subprocess.run([str(binary), str(bin_path), "--logits", "--prompt",
                        prompt.decode("latin-1"), "--max-ctx", str(max_ctx)],
                       capture_output=True, text=True, check=True)
    vals = [float(x) for x in r.stdout.split()]
    assert len(vals) == 256
    return np.array(vals)


def _oracle_logits(model, prompt: bytes, max_ctx=96):
    ids = list(prompt)[-max_ctx:] or [32]
    logits, _ = model.forward(np.array([ids]))
    return logits.data[0, -1]


PROMPTS = [b"Hello world, this is a test.",
           b"def add(a, b):\n    return a",
           b"words and punctuation, mixed! ok?"]


def test_dense_logits_parity_random_model(binary, tmp_path):
    m = _small_model()
    bin_path = tmp_path / "m.bin"
    _export(m, bin_path)
    for prompt in PROMPTS:
        fast = _cpp_logits(binary, bin_path, prompt)
        dense = _oracle_logits(m, prompt)
        assert np.allclose(fast, dense, atol=1e-8), np.abs(fast - dense).max()
        assert int(fast.argmax()) == int(dense.argmax())


def test_greedy_generation_byte_identical(binary, tmp_path):
    m = _small_model(11)
    bin_path = tmp_path / "m.bin"
    _export(m, bin_path)
    for prompt in PROMPTS:
        r = subprocess.run([str(binary), str(bin_path), "--prompt",
                            prompt.decode("latin-1"), "--max-new", "24",
                            "--max-ctx", "96", "--hex"],
                           capture_output=True, text=True, check=True)
        fast = bytes.fromhex(r.stdout.strip())
        dense = m.generate(prompt, max_new=24, greedy=True, max_ctx=96)
        assert fast == dense, (prompt, fast, dense)


def test_greedy_identical_through_window_slide(binary, tmp_path):
    m = _small_model(13)
    bin_path = tmp_path / "m.bin"
    _export(m, bin_path)
    prompt = b"The quick brown fox jumps over the lazy dog. " * 2
    r = subprocess.run([str(binary), str(bin_path), "--prompt",
                        prompt.decode("latin-1"), "--max-new", "30",
                        "--max-ctx", "40", "--hex"],
                       capture_output=True, text=True, check=True)
    fast = bytes.fromhex(r.stdout.strip())
    dense = m.generate(prompt, max_new=30, greedy=True, max_ctx=40)
    assert fast == dense


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="trained checkpoint absent")
def test_real_checkpoint_parity(binary, tmp_path):
    from cfna.microtorch.chat import load_checkpoint
    model, _ = load_checkpoint(str(REAL_CKPT))
    bin_path = tmp_path / "real.bin"
    _export(model, bin_path)
    prompt = b"User: Hello\nAssistant: "
    fast = _cpp_logits(binary, bin_path, prompt, max_ctx=288)
    dense = _oracle_logits(model, prompt, max_ctx=288)
    assert np.allclose(fast, dense, atol=1e-8), np.abs(fast - dense).max()
    assert int(fast.argmax()) == int(dense.argmax())
