"""Retrieval is wired into training and NueronceModel learns to use it —
mirrors tests/test_retrieval_training.py. No PyTorch needed."""

import numpy as np

from nueronce.engine import retrieval_train as rt
from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig


def _cfg():
    return NueronceConfig(byte_embed_dim=16, d_local=24, d_model=32, p_max=12,
                            physical_blocks=1, logical_depth=2, n_heads=4, unit_window=10,
                            decoder_window=14, decoder_layers=1, d_state=6, channel_dim=8,
                            min_patch=2, max_patch=10)


def test_retrieval_batch_shapes_and_recall():
    rng = np.random.default_rng(0)
    b = rt.make_retrieval_batch(rng, batch=16, k=2)
    assert b["seq_ids"].shape == (16, rt.SEQ_TEMPLATE_LEN)
    assert b["neighbor_ids"].shape[0] == 16 and b["neighbor_ids"].shape[1] == 2
    assert b["value_mask"][:, rt.VALUE_POS].all()
    assert b["recall_at_k"] == 1.0          # frozen retriever finds the matching fact


def test_retrieval_path_receives_gradients():
    np.random.seed(0)
    m = NueronceModel(_cfg())
    rng = np.random.default_rng(0)
    b = rt.make_retrieval_batch(rng, batch=8, k=2)
    logits, _ = m.forward(b["seq_ids"], b["neighbor_ids"], b["neighbor_mask"])
    loss = m.masked_token_loss(logits, b["seq_ids"], b["value_mask"])
    loss.backward()
    named = {
        "ret_proj.weight": m.ret_proj.weight,
        "ret_byte_embed.weight": m.ret_byte_embed.weight,
        "decoder.layers[0].ret_attn.q.weight": m.decoder.layers[0].ret_attn.q.weight,
        "core.blocks[0].retrieval.q.weight": m.core.blocks[0].retrieval.q.weight,
    }
    for name, p in named.items():
        assert p.grad is not None and float(np.abs(p.grad).sum()) > 0, f"no gradient into {name}"


def test_retrieval_improves_value_prediction():
    np.random.seed(0)
    m = NueronceModel(_cfg())
    hist = rt.train_retrieval(m, steps=250, batch=24, k=2, lr=5e-3, seed=0, log_every=249)
    final = hist[-1]
    # With retrieval the value is predictable; without it, it cannot be (fresh
    # random per example). Expect a clear gap in both loss and accuracy.
    assert final["val_without_retrieval"] > final["val_with_retrieval"] + 0.5
    assert final["acc_with"] > final["acc_without"] + 0.2
    assert final["acc_with"] > 0.30          # well above 0.10 chance for 10 values
