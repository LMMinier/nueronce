"""Required tests from FOUNDATIONAL_GENERATION_RECOVERY.md, the parts that
don't need a trained checkpoint or GPU time (fast, deterministic, CI-safe):

  1. training and inference prompt-prefix byte equality
  2. response mask beginning at the correct byte
  3. target shifted exactly one byte (masked_token_loss indexing)
  4. EOS/terminator bytes are inside the training loss mask
  5. no partial delimiter leakage (STOP_SEQUENCES catches "<|")
  6. state isolation between prompts (generate() carries no hidden state)
  7. checkpoint architecture agreement (config mismatch fails loudly)

Item 8 (exact-overfit reproduction, 31/32) and item 9 (dense vs incremental
agreement) are exercised end-to-end by scripts/train_tiny_exact_overfit.py +
scripts/eval_tiny_exact_overfit.py and tests/test_incremental_torch.py
respectively -- both take real training time, so they are not duplicated
here. Item 10 (proof gate unchanged) is enforced by not editing
scripts/eval_foundational_proof_gate.py or its 8 CASES.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nueronce.model import ModelConfig, NUERONCEModel
from nueronce.prompting import ASSISTANT, END, STOP_SEQUENCES, format_inference_prompt, format_training_example
from nueronce.training.dialogue_data import encode_example, make_sft_batch


def _tiny_config(**overrides) -> ModelConfig:
    base = dict(byte_embed_dim=8, d_local=12, d_model=16, p_max=8, physical_blocks=1,
               logical_depth=2, n_heads=2, unit_window=6, decoder_window=8,
               decoder_layers=2, d_state=4, channel_dim=4, min_patch=2, max_patch=6)
    base.update(overrides)
    return ModelConfig(**base)


# -- 1. training and inference prompt-prefix byte equality ------------------

def test_training_prefix_matches_inference_prompt_bytes():
    system, prompt = "You are CFNA.", "What is 2 + 2?"
    inference_prompt = format_inference_prompt(
        system_message=system, user_request=prompt, trusted_evidence="", response_plan="")
    train_bytes, mask = format_training_example(
        system_message=system, user_request=prompt, assistant_response="4")
    prefix_len = len(inference_prompt.encode("utf-8"))
    assert train_bytes[:prefix_len] == inference_prompt.encode("utf-8"), (
        "the bytes a trainer masks out as 'prompt' must be byte-identical to "
        "what format_inference_prompt renders at generation time, or the "
        "model is trained on one prefix and evaluated on another")
    # everything before the prefix boundary must be unmasked (loss-free)
    assert not any(mask[:prefix_len])


def test_encode_example_prefix_matches_inference_prompt():
    """The actual trainer path (dialogue_data.encode_example, used by
    scripts/train_forgeloop_sft.py) must agree with the actual eval path
    (prompting.format_inference_prompt, used by
    scripts/eval_foundational_proof_gate.py) on the exact same prefix."""
    system, prompt = "You are CFNA.", "Rewrite politely: Send it now."
    train_bytes, mask = encode_example(prompt, "Could you please send it now?", system=system)
    inference_prompt = format_inference_prompt(
        system_message=system, user_request=prompt, trusted_evidence="", response_plan="")
    prefix_len = len(inference_prompt.encode("utf-8"))
    assert train_bytes[:prefix_len] == inference_prompt.encode("utf-8")
    assert not any(mask[:prefix_len])


# -- 2. response mask begins at the correct byte -----------------------------

def test_mask_starts_exactly_at_assistant_content():
    system, prompt, response = "Sys.", "Hi", "Hello there"
    full, mask = format_training_example(
        system_message=system, user_request=prompt, assistant_response=response)
    prefix = format_inference_prompt(system_message=system, user_request=prompt,
                                     trusted_evidence="", response_plan="")
    prefix_bytes = prefix.encode("utf-8")
    assert full[:len(prefix_bytes)] == prefix_bytes
    first_true = mask.index(True)
    assert first_true == len(prefix_bytes) == len(prefix_bytes)
    assert full[first_true:first_true + len(response)] == response.encode("utf-8")


# -- 3. target shifted exactly one byte (masked_token_loss indexing) --------

def test_masked_token_loss_predicts_next_byte_not_current():
    """logits[:, t] must be scored against byte_ids[:, t+1], not byte_ids[:, t].
    Verified by handing masked_token_loss a logits tensor that is a perfect
    predictor of byte_ids shifted by one, and asserting near-zero loss; and a
    tensor that perfectly predicts byte_ids *unshifted*, asserting much higher
    loss (it must NOT reward same-position copying)."""
    model = NUERONCEModel(_tiny_config())
    byte_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    mask = torch.tensor([[False, False, True, True, True]])

    perfect_next = torch.full((1, 5, 256), -20.0)
    for t in range(4):
        perfect_next[0, t, byte_ids[0, t + 1]] = 20.0
    loss_shifted = model.masked_token_loss(perfect_next, byte_ids, mask)
    assert float(loss_shifted) < 1e-3

    perfect_same = torch.full((1, 5, 256), -20.0)
    for t in range(5):
        perfect_same[0, t, byte_ids[0, t]] = 20.0
    loss_unshifted = model.masked_token_loss(perfect_same, byte_ids, mask)
    assert float(loss_unshifted) > 5.0


# -- 4. EOS/terminator bytes are inside the training loss mask --------------

def test_end_marker_is_inside_the_loss_mask():
    full, mask = format_training_example(
        system_message="Sys.", user_request="Hi", assistant_response="Hello")
    end_bytes = END.encode("utf-8")
    idx = full.rfind(end_bytes)
    assert idx != -1
    assert all(mask[idx:idx + len(end_bytes)]), (
        "the model must be trained to predict its own stop marker, or "
        "generation will never learn to terminate cleanly")


# -- 5. no partial delimiter leakage -----------------------------------------

def test_stop_sequences_cover_bare_delimiter_open():
    assert b"<|" in STOP_SEQUENCES, (
        "STOP_SEQUENCES must catch a bare '<|' so a malformed/partial tag "
        "(e.g. '<|en' cut short) cannot leak into a user-facing answer")


@torch.no_grad()
def test_generate_strips_matched_stop_sequence_from_output():
    """A model that is forced (via monkeypatched forward) to immediately
    emit the two bytes of '<|' must return zero generated bytes, not a
    dangling '<|' fragment."""
    model = NUERONCEModel(_tiny_config())
    open_tag = b"<|"
    call_count = [0]

    def fake_forward(ctx, *a, **k):
        t = ctx.shape[1]
        logits = torch.full((1, t, 256), -20.0)
        next_byte = open_tag[min(call_count[0], len(open_tag) - 1)]
        call_count[0] += 1
        logits[0, -1, next_byte] = 20.0
        return logits, None

    model.forward = fake_forward  # type: ignore[method-assign]
    out = model.generate(b"prompt", max_new=8, temperature=0.0, greedy=True,
                         max_ctx=64, stop_sequences=STOP_SEQUENCES, continuation_only=True)
    assert b"<|" not in out
    assert len(out) == 0


# -- 6. state isolation between prompts --------------------------------------

@torch.no_grad()
def test_generate_has_no_cross_call_state_leak():
    """Calling generate() on prompt B after prompt A must not change prompt
    A's result -- there must be no hidden module-level state carried between
    independent generate() calls."""
    model = NUERONCEModel(_tiny_config())
    model.eval()
    prompt_a = format_inference_prompt(
        system_message="Sys.", user_request="First question", trusted_evidence="", response_plan=""
    ).encode("utf-8")
    prompt_b = format_inference_prompt(
        system_message="Sys.", user_request="Second, unrelated question", trusted_evidence="", response_plan=""
    ).encode("utf-8")

    first_a = model.generate(prompt_a, max_new=16, temperature=0.0, greedy=True,
                             max_ctx=256, stop_sequences=STOP_SEQUENCES, continuation_only=True)
    _ = model.generate(prompt_b, max_new=16, temperature=0.0, greedy=True,
                       max_ctx=256, stop_sequences=STOP_SEQUENCES, continuation_only=True)
    second_a = model.generate(prompt_a, max_new=16, temperature=0.0, greedy=True,
                              max_ctx=256, stop_sequences=STOP_SEQUENCES, continuation_only=True)
    assert first_a == second_a


# -- 7. checkpoint architecture agreement ------------------------------------

def test_loading_mismatched_architecture_fails_loudly(tmp_path):
    """A checkpoint's saved config must exactly reconstruct the architecture
    that produced its state_dict -- loading it into a *different* config must
    raise, not silently produce garbage from mismatched tensor shapes."""
    small = NUERONCEModel(_tiny_config())
    bigger = _tiny_config(d_model=32)  # deliberately incompatible d_model
    with pytest.raises((RuntimeError, ValueError)):
        NUERONCEModel(bigger).load_state_dict(small.state_dict())
