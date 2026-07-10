"""Retrieval is wired into training and the model learns to use it."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nueronce import retrieval_train as rt
from nueronce.model import NUERONCEModel, ModelConfig


def _cfg():
    return ModelConfig(byte_embed_dim=32, d_local=64, d_model=96, p_max=16,
                       physical_blocks=1, logical_depth=2, n_heads=4, unit_window=16,
                       decoder_window=24, decoder_layers=2, d_state=8, channel_dim=16)


def test_retrieval_batch_shapes_and_recall():
    rng = np.random.default_rng(0)
    b = rt.make_retrieval_batch(rng, batch=16, k=2)
    assert b["seq_ids"].shape == (16, rt.SEQ_TEMPLATE_LEN)
    assert b["neighbor_ids"].shape[0] == 16 and b["neighbor_ids"].shape[1] == 2
    assert b["value_mask"][:, rt.VALUE_POS].all()
    assert b["recall_at_k"] == 1.0          # frozen retriever finds the matching fact


def test_retrieval_path_receives_gradients():
    torch.manual_seed(0)
    m = NUERONCEModel(_cfg())
    rng = np.random.default_rng(0)
    b = rt.make_retrieval_batch(rng, batch=8, k=2)
    logits, _ = m(b["seq_ids"], b["neighbor_ids"], b["neighbor_mask"])
    loss = m.masked_token_loss(logits, b["seq_ids"], b["value_mask"])
    loss.backward()
    named = dict(m.named_parameters())
    for name in ["ret_proj.weight", "ret_byte_embed.weight",
                 "decoder.layers.0.ret_attn.q.weight", "core.blocks.0.retrieval.q.weight"]:
        g = named[name].grad
        assert g is not None and float(g.abs().sum()) > 0, f"no gradient into {name}"


def test_retrieval_improves_value_prediction():
    torch.manual_seed(0)
    m = NUERONCEModel(_cfg())
    hist = rt.train_retrieval(m, steps=200, batch=24, k=2, lr=4e-3, seed=0, log_every=199)
    final = hist[-1]
    # With retrieval the value is predictable; without it, it cannot be (fresh
    # random per example). Expect a clear gap in both loss and accuracy.
    assert final["val_without_retrieval"] > final["val_with_retrieval"] + 0.5
    assert final["acc_with"] > final["acc_without"] + 0.2
    assert final["acc_with"] > 0.30          # well above 0.10 chance for 10 values
