"""VGRFT stage 1 (SFT) on the from-scratch Nueronce Engine.

No PyTorch import anywhere in this file (or in what it imports) — it proves
the SFT training pass is real using nothing but the hand-built engine
autograd engine (see nueronce/engine/), so it also runs in environments where
PyTorch isn't installed.
"""

import numpy as np

from nueronce.engine import functional as F
from nueronce.engine.models import MicroByteLM, MicroSFTBackend, train_dialogue_sft
from nueronce.training.dialogue_data import (
    BOT_TAG, SFT_DATASET, USER_TAG, encode_example, held_out_split, make_sft_batch,
)
from nueronce.training.vgrft import VGRFTTrainer
from nueronce.prompting import END, format_inference_prompt

TOY = [
    ("Hello", "Hi there!"),
    ("Thank you", "You are welcome!"),
    ("Goodbye", "Bye!"),
    ("What is two plus two?", "Four."),
]


def test_encode_example_masks_only_the_response_and_its_stop_byte():
    full, mask = encode_example("Hello", "Hi!")
    assert len(full) == len(mask)
    prompt = format_inference_prompt(
        system_message="", user_request="Hello", trusted_evidence="", response_plan=""
    )
    assert full.decode("utf-8") == prompt + f"Hi!\n{END}\n"
    prefix_len = len(prompt.encode("utf-8"))
    assert not any(mask[:prefix_len])
    assert all(mask[prefix_len:])


def test_make_sft_batch_shapes_and_padding_mask():
    batch = make_sft_batch(TOY)
    b, t = batch["byte_ids"].shape
    assert b == len(TOY)
    assert batch["byte_ids"].dtype == np.int64
    assert batch["target_mask"].dtype == np.bool_
    lengths = [len(encode_example(p, r)[0]) for p, r in TOY]
    for i, length in enumerate(lengths):
        assert not batch["target_mask"][i, length:].any()


def test_held_out_split_is_disjoint_and_covers_all_examples():
    train, val = held_out_split(SFT_DATASET, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(SFT_DATASET)
    assert set(train).isdisjoint(set(val))


def test_masked_cross_entropy_matches_manual_row_filter():
    from nueronce.engine.tensor import Tensor
    rng = np.random.default_rng(0)
    logits = Tensor(rng.normal(size=(6, 5)))
    targets = rng.integers(0, 5, size=6)
    mask = np.array([True, False, True, False, False, True])

    got = F.masked_cross_entropy(logits, targets, mask)
    idx = np.nonzero(mask)[0]
    want = F.cross_entropy(logits[idx], targets[idx])
    assert abs(got.item() - want.item()) < 1e-9


def test_masked_loss_matches_manual_single_position_cross_entropy():
    from nueronce.engine.tensor import Tensor, no_grad

    model = MicroByteLM(d_model=8, n_heads=2, window=4, d_state=4)
    ids = np.array([[5, 10, 15, 20, 25, 30]])
    mask = np.zeros_like(ids, dtype=bool)
    mask[:, 4] = True   # only the target at ids[:, 4] should count

    with no_grad():
        logits = model.forward(ids[:, :-1])          # [1, 5, 256]
    row = logits.data[0, 3]                            # predicts ids[:, 4]
    want = F.cross_entropy(Tensor(row.reshape(1, -1)), np.array([ids[0, 4]])).item()

    got = model.masked_loss(ids, mask).item()
    assert abs(got - want) < 1e-6


def test_micro_dialogue_sft_loss_decreases_on_a_toy_set():
    np.random.seed(0)
    model = MicroByteLM(d_model=16, n_heads=2, window=8, d_state=6)

    first_batch = make_sft_batch(TOY)
    first_loss = model.masked_loss(first_batch["byte_ids"], first_batch["target_mask"]).item()

    history = train_dialogue_sft(model, TOY, steps=150, batch_size=4, lr=1e-2, seed=0, log_every=150)
    last_loss = model.masked_loss(first_batch["byte_ids"], first_batch["target_mask"]).item()

    assert history and "train_loss" in history[-1]
    assert last_loss < 0.6 * first_loss, f"did not learn: {first_loss:.3f} -> {last_loss:.3f}"


def test_vgrft_supervised_instruction_tune_runs_with_engine_backend():
    np.random.seed(0)
    model = MicroByteLM(d_model=16, n_heads=2, window=8, d_state=6)
    trainer = VGRFTTrainer(MicroSFTBackend(model, lr=1e-2))

    history = trainer.supervised_instruction_tune(TOY, steps=20, batch_size=4, log_every=20)
    assert history and history[-1]["step"] == 20
