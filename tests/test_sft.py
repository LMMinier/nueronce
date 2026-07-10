"""VGRFT stage 1 (supervised instruction tuning) is real: it actually trains."""

import pytest

torch = pytest.importorskip("torch")

from nueronce.chat import Conversation
from nueronce.model import NUERONCEModel, ModelConfig
from nueronce.training.sft import (
    SFT_DATASET, encode_example, make_sft_batch, held_out_split,
    sft_step, sft_eval, train_sft, TorchSFTBackend, USER_TAG, BOT_TAG,
)
from nueronce.prompting import END, format_inference_prompt
from nueronce.training.vgrft import VGRFTTrainer


def _tiny() -> NUERONCEModel:
    return NUERONCEModel(ModelConfig(
        byte_embed_dim=16, d_local=32, d_model=48, p_max=16, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8))


TOY = [
    ("Hello", "Hi there!"),
    ("Thank you", "You are welcome!"),
    ("Goodbye", "Bye!"),
    ("What is two plus two?", "Four."),
]


def test_tags_match_conversation_layout():
    assert USER_TAG == Conversation.user_tag
    assert BOT_TAG == Conversation.bot_tag


def test_encode_example_masks_only_the_response_and_its_stop_byte():
    full, mask = encode_example("Hello", "Hi!")
    assert len(full) == len(mask)
    text = full.decode("utf-8")
    assert text == format_inference_prompt(
        system_message="", user_request="Hello", trusted_evidence="", response_plan=""
    ) + f"Hi!\n{END}\n"
    prefix_len = len(format_inference_prompt(
        system_message="", user_request="Hello", trusted_evidence="", response_plan=""
    ).encode("utf-8"))
    assert not any(mask[:prefix_len])
    assert all(mask[prefix_len:])
    assert full.decode("utf-8")[prefix_len:] == f"Hi!\n{END}\n"


def test_make_sft_batch_shapes_and_padding_mask():
    batch = make_sft_batch(TOY)
    b, t = batch["byte_ids"].shape
    assert b == len(TOY)
    assert batch["target_mask"].shape == (b, t)
    # shorter examples are right-padded with mask False beyond their real length
    lengths = [len(encode_example(p, r)[0]) for p, r in TOY]
    for i, length in enumerate(lengths):
        assert not batch["target_mask"][i, length:].any()


def test_held_out_split_is_disjoint_and_covers_all_examples():
    train, val = held_out_split(SFT_DATASET, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(SFT_DATASET)
    assert set(train).isdisjoint(set(val))


def test_sft_loss_decreases_on_a_toy_dialogue_set():
    torch.manual_seed(0)
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    first_batch = make_sft_batch(TOY)
    first_loss = sft_eval(model, first_batch)

    history = train_sft(model, opt, TOY, steps=150, batch_size=4, seed=0, log_every=150)
    last_loss = sft_eval(model, first_batch)

    assert history and "train_loss" in history[-1]
    assert last_loss < 0.5 * first_loss, f"did not learn: {first_loss:.3f} -> {last_loss:.3f}"


def test_vgrft_supervised_instruction_tune_runs_with_torch_backend():
    torch.manual_seed(0)
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    trainer = VGRFTTrainer(TorchSFTBackend(model, opt))

    history = trainer.supervised_instruction_tune(TOY, steps=20, batch_size=4, log_every=20)
    assert history and history[-1]["step"] == 20


def test_vgrft_stage1_still_raises_without_a_compatible_backend():
    with pytest.raises(NotImplementedError):
        VGRFTTrainer().supervised_instruction_tune(TOY)
    with pytest.raises(NotImplementedError):
        VGRFTTrainer(backend=object()).supervised_instruction_tune(TOY)


def test_vgrft_other_stages_are_still_unimplemented():
    trainer = VGRFTTrainer(TorchSFTBackend(_tiny(), None))
    for stage, arg in [("tool_grounded_tune", []), ("verifier_train", []), ("residual_expert_train", [])]:
        with pytest.raises(NotImplementedError):
            getattr(trainer, stage)(arg)
