"""NueronceModel — the real NUERONCE architecture (not a smaller stand-in) ported
onto the from-scratch Nueronce Engine. No PyTorch import anywhere in this
file or what it imports, so it runs wherever only NumPy is available.

Mirrors tests/test_model_learns.py's bar for the real torch model: forward
shapes, exact causality, and a from-scratch learning curve.
"""

import numpy as np

from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.engine.models import MicroSFTBackend
from nueronce.engine.optim import AdamW, clip_grad_norm_
from nueronce.engine.tensor import no_grad
from nueronce.training.dialogue_data import held_out_split, SFT_DATASET
from nueronce.training.vgrft import VGRFTTrainer


def _tiny() -> NueronceConfig:
    return NueronceConfig(
        byte_embed_dim=12, d_local=16, d_model=24, p_max=12, physical_blocks=1,
        logical_depth=2, n_heads=4, unit_window=8, decoder_window=12,
        decoder_layers=1, d_state=6, channel_dim=8, ret_byte_dim=8,
        min_patch=2, max_patch=10,
    )


def test_forward_and_loss_shapes():
    np.random.seed(0)
    m = NueronceModel(_tiny())
    ids = np.random.randint(0, 256, size=(3, 40))
    logits, boundary = m.forward(ids)
    assert logits.shape == (3, 40, 256)
    assert boundary.shape == (3, 40)
    loss, parts = m.loss(ids)
    assert loss.requires_grad
    assert {"loss", "lm", "boundary", "bpb"} <= set(parts)


def test_model_is_causal():
    np.random.seed(0)
    m = NueronceModel(_tiny())
    ids = np.random.randint(0, 256, size=(1, 40))
    with no_grad():
        base, _ = m.forward(ids)
        pert = ids.copy()
        pert[0, 30] = (pert[0, 30] + 111) % 256
        out, _ = m.forward(pert)
    # editing a future byte must not change any earlier position's logits
    assert np.abs(base.data[0, :30] - out.data[0, :30]).max() < 1e-8


def test_retrieval_path_only_gets_gradient_when_used():
    np.random.seed(0)
    m = NueronceModel(_tiny())
    ids = np.random.randint(0, 256, size=(2, 20))
    loss, _ = m.loss(ids)
    loss.backward()
    # no neighbor_ids passed -> retrieval-only params never entered the graph
    assert m.ret_byte_embed.weight.grad is None

    m2 = NueronceModel(_tiny())
    neighbor_ids = np.random.randint(0, 256, size=(2, 3, 10))
    logits, _ = m2.forward(ids, neighbor_ids)
    target_mask = np.zeros_like(ids, dtype=bool)
    target_mask[:, 5] = True
    m2.masked_token_loss(logits, ids, target_mask).backward()
    assert m2.ret_byte_embed.weight.grad is not None


def test_num_params_matches_parameter_count():
    m = NueronceModel(_tiny())
    assert m.num_params() == sum(p.data.size for p in m.parameters())
    assert m.num_params() > 0


def test_generate_runs_and_returns_bytes():
    np.random.seed(0)
    m = NueronceModel(_tiny())
    out = m.generate(b"hi", max_new=8, greedy=True)
    assert isinstance(out, bytes)
    assert out[:2] == b"hi"
    assert len(out) == 10


def test_model_learns_on_toy_corpus():
    np.random.seed(0)
    m = NueronceModel(_tiny())
    opt = AdamW(list(m.parameters()), lr=3e-3)
    text = b"the quick brown fox jumps over the lazy dog. " * 8
    data = np.frombuffer(text, dtype=np.uint8)
    rng = np.random.default_rng(0)

    def batch(seq_len=48, bs=4):
        idx = rng.integers(0, len(data) - seq_len, size=bs)
        return np.stack([data[i:i + seq_len] for i in idx]).astype(np.int64)

    first = last = None
    for _ in range(120):
        loss, parts = m.loss(batch())
        m.zero_grad()
        loss.backward()
        clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = first if first is not None else parts["lm"]
        last = parts["lm"]

    assert last < 0.3 * first, f"did not learn: {first:.3f} -> {last:.3f}"
    assert last < 3.0, f"loss not below uniform baseline: {last:.3f}"


def test_full_model_runs_vgrft_sft_via_microsftbackend():
    """The ported architecture, not just MicroByteLM, is a drop-in VGRFT
    stage-1 backend: same duck-typed .masked_loss/.parameters()/.zero_grad()."""
    np.random.seed(0)
    model = NueronceModel(_tiny())
    trainer = VGRFTTrainer(MicroSFTBackend(model, lr=5e-3))

    toy = [("Hello", "Hi there!"), ("Thank you", "You are welcome!"), ("Goodbye", "Bye!")]
    history = trainer.supervised_instruction_tune(toy, steps=60, batch_size=3, log_every=20)
    assert history[0]["train_loss"] > history[-1]["train_loss"]


def test_held_out_split_still_disjoint_for_full_model_sft():
    train, val = held_out_split(SFT_DATASET, val_frac=0.2, seed=0)
    assert set(train).isdisjoint(set(val))
