"""The architecture runs, is causal, and actually learns.

A small/fast configuration so the suite stays quick; the full demo lives in
scripts/train_demo.py.
"""

import pytest

torch = pytest.importorskip("torch")

from cfna.data import corpus_bytes, make_batches
from cfna.model import CFNAModel, ModelConfig


def _tiny() -> ModelConfig:
    return ModelConfig(
        byte_embed_dim=24, d_local=48, d_model=64, p_max=24,
        physical_blocks=1, logical_depth=2, n_heads=4, unit_window=16,
        decoder_window=24, decoder_layers=1, d_state=8, channel_dim=16,
        min_patch=3, max_patch=20,
    )


def test_forward_and_loss_shapes():
    m = CFNAModel(_tiny())
    ids = torch.randint(0, 256, (3, 64))
    logits, boundary = m(ids)
    assert logits.shape == (3, 64, 256)
    assert boundary.shape == (3, 64)
    loss, parts = m.loss(ids)
    assert loss.requires_grad
    assert {"loss", "lm", "boundary", "bpb"} <= set(parts)


def test_model_is_causal():
    torch.manual_seed(0)
    m = CFNAModel(_tiny()).eval()
    ids = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        base, _ = m(ids)
        pert = ids.clone(); pert[0, 48] = (pert[0, 48] + 9) % 256
        out, _ = m(pert)
    # editing a future byte must not change any earlier position's logits
    assert (base[0, :48] - out[0, :48]).abs().max() < 1e-5


def test_model_learns_on_toy_corpus():
    torch.manual_seed(0)
    m = CFNAModel(_tiny())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    data = corpus_bytes(repeat=6)
    batches = make_batches(data, seq_len=64, batch_size=8, n_batches=60, seed=0)

    first = None
    last = None
    for batch in batches:
        loss, parts = m.loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = first if first is not None else parts["lm"]
        last = parts["lm"]

    # The LM loss should fall well below both its start and the uniform baseline
    # (ln 256 ~= 5.545). A loose threshold keeps the test robust.
    assert last < 0.6 * first, f"did not learn: {first:.3f} -> {last:.3f}"
    assert last < 3.0, f"loss not below uniform baseline: {last:.3f}"
