"""Guard against the prompt-format drift that silently broke legacy checkpoints.

When the shared dialogue tags were repointed from "User: "/"Assistant: " to the
canonical <|user|>/<|assistant|> markers, every checkpoint trained on the old
format started scoring ~0.52 byte-accuracy instead of ~0.91 at inference — a
format mismatch, not a weak model. These tests pin the two invariants that
prevent a silent recurrence. No torch needed.
"""

import numpy as np

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from cfna.microtorch.chat import MicroConversation
from cfna.training.sharded_sft import save_checkpoint
from cfna.microtorch.optim import AdamW


def test_legacy_context_is_byte_exact_training_prefix():
    convo = MicroConversation(model=None, prompt_format="legacy")
    assert convo._context("Hello") == b"User: Hello\nAssistant: "


def test_canonical_context_uses_markers():
    convo = MicroConversation(model=None, prompt_format="canonical")
    assert convo._context("Hello") == b"<|system|>\n\n<|user|>\nHello\n<|assistant|>\n"


def test_resolve_format_defaults_legacy_for_old_checkpoints():
    # no meta / no prompt_format -> legacy (checkpoints predating the stamp)
    assert MicroConversation.resolve_format({}) == "legacy"
    assert MicroConversation.resolve_format({"meta": {}}) == "legacy"
    assert MicroConversation.resolve_format({"meta": {"prompt_format": "canonical"}}) == "canonical"


def test_new_checkpoints_stamp_prompt_format(tmp_path):
    from cfna.training.dialogue_data import PROMPT_FORMAT
    m = MicroCFNAModel(MicroModelConfig(
        byte_embed_dim=8, d_local=12, d_model=16, p_max=8, physical_blocks=1,
        logical_depth=1, n_heads=2, unit_window=6, decoder_window=8,
        decoder_layers=1, d_state=4, channel_dim=4, min_patch=2, max_patch=8))
    opt = AdamW(list(m.parameters()), lr=1e-3)
    path = tmp_path / "ck.pt"
    save_checkpoint(str(path), m, opt, {"global_step": 1})
    import pickle
    payload = pickle.load(open(path, "rb"))
    assert payload["meta"]["prompt_format"] == PROMPT_FORMAT
    assert MicroConversation.resolve_format(payload) == PROMPT_FORMAT
