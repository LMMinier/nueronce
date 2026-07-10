"""Held-out evaluation harness + matched baselines on engine (reviewer
Priority 1), mirroring tests/test_eval_harness.py. No PyTorch needed."""

import numpy as np

from nueronce.engine.baselines import BaselineConfig, ByteSSMLM, ByteTransformerLM
from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.engine.data import larger_corpus_bytes, make_batches, train_val_split
from nueronce.engine.eval_harness import compare
from nueronce.engine.tensor import no_grad


def test_train_val_split_is_disjoint():
    data = larger_corpus_bytes()
    tr, va = train_val_split(data, val_frac=0.25)
    assert len(tr) + len(va) == len(data)
    assert tr == data[: len(tr)] and va == data[len(tr):]   # disjoint regions
    assert len(va) > 64


def test_baselines_train_and_are_causal():
    m = ByteTransformerLM(BaselineConfig(d_model=32, n_layers=2, n_heads=4, max_len=24))
    ids = np.random.randint(0, 256, size=(2, 24))
    loss = m.lm_loss(ids)
    m.zero_grad()
    loss.backward()
    assert np.isfinite(loss.item())
    with no_grad():
        base = m.forward(ids)
        pert = ids.copy(); pert[0, 15] = (pert[0, 15] + 5) % 256
        out = m.forward(pert)
    assert np.abs(base.data[0, :15] - out.data[0, :15]).max() < 1e-8  # causal


def test_compare_reports_heldout_for_all():
    tr, va = train_val_split(larger_corpus_bytes(), 0.25)
    train_batches = make_batches(tr, 24, 4, 12, seed=0)
    val_batches = make_batches(va, 24, 4, 3, seed=1)
    factories = {
        "NUERONCE": lambda: NueronceModel(NueronceConfig(
            byte_embed_dim=12, d_local=16, d_model=24, p_max=8, physical_blocks=1,
            logical_depth=2, n_heads=4, unit_window=8, decoder_window=12,
            decoder_layers=1, d_state=6, channel_dim=8, min_patch=2, max_patch=8)),
        "Transformer": lambda: ByteTransformerLM(BaselineConfig(d_model=24, n_layers=2, max_len=24)),
        "SSM": lambda: ByteSSMLM(BaselineConfig(d_model=24, n_layers=2, d_state=6, max_len=24)),
    }
    res = compare(factories, train_batches, val_batches)
    for name, r in res.items():
        assert r["params"] > 0
        assert 0.0 < r["heldout_bpb"] < 9.0, f"{name} heldout bpb out of range"
        assert r["heldout_bpb"] != r["final_train_bpb"]
