"""Held-out evaluation harness + matched baselines (reviewer Priority 1)."""

import pytest

torch = pytest.importorskip("torch")

from cfna.baselines import BaselineConfig, ByteSSMLM, ByteTransformerLM
from cfna.data import larger_corpus_bytes, make_batches, train_val_split
from cfna.eval import bits_per_byte, compare
from cfna.model import CFNAModel, ModelConfig


def test_train_val_split_is_disjoint():
    data = larger_corpus_bytes()
    tr, va = train_val_split(data, val_frac=0.25)
    assert len(tr) + len(va) == len(data)
    assert tr == data[: len(tr)] and va == data[len(tr):]   # disjoint regions
    assert len(va) > 64


def test_baselines_train_and_are_causal():
    torch.manual_seed(0)
    m = ByteTransformerLM(BaselineConfig(d_model=64, n_layers=2, n_heads=4, max_len=48))
    ids = torch.randint(0, 256, (2, 48))
    loss = m.lm_loss(ids)
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    with torch.no_grad():
        base = m(ids)
        pert = ids.clone(); pert[0, 30] = (pert[0, 30] + 5) % 256
        out = m(pert)
    assert (base[0, :30] - out[0, :30]).abs().max() < 1e-5  # causal


def test_compare_reports_heldout_for_all():
    tr, va = train_val_split(larger_corpus_bytes(), 0.25)
    train_batches = make_batches(tr, 48, 8, 30, seed=0)
    val_batches = make_batches(va, 48, 8, 3, seed=1)
    factories = {
        "CFNA": lambda: CFNAModel(ModelConfig(
            byte_embed_dim=24, d_local=48, d_model=64, p_max=16, physical_blocks=1,
            logical_depth=2, n_heads=4, unit_window=12, decoder_window=16,
            decoder_layers=1, d_state=8, channel_dim=12)),
        "Transformer": lambda: ByteTransformerLM(BaselineConfig(d_model=64, n_layers=2, max_len=48)),
        "SSM": lambda: ByteSSMLM(BaselineConfig(d_model=64, n_layers=2, d_state=8, max_len=48)),
    }
    res = compare(factories, train_batches, val_batches)
    for name, r in res.items():
        assert r["params"] > 0
        assert 0.0 < r["heldout_bpb"] < 9.0, f"{name} heldout bpb out of range"
        # held-out should differ from training loss (not the same data)
        assert r["heldout_bpb"] != r["final_train_bpb"]
