"""Chat interface wiring (uses a tiny untrained model — output quality not asserted)."""

import pytest

torch = pytest.importorskip("torch")

from nueronce.chat import Conversation, load_checkpoint
from nueronce.model import NUERONCEModel, ModelConfig


def _tiny():
    return NUERONCEModel(ModelConfig(
        byte_embed_dim=16, d_local=32, d_model=48, p_max=16, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8))


def test_conversation_turn_returns_text_and_tracks_transcript():
    convo = Conversation(_tiny(), temperature=0.7, max_new=12, min_new=2)
    reply = convo.say("Hello there, who are you?")
    assert isinstance(reply, str)
    assert len(convo.transcript) == 2            # user + assistant recorded
    assert convo.transcript[0] == ("user", "Hello there, who are you?")


def test_checkpoint_roundtrip(tmp_path):
    m = _tiny()
    cfg_dict = vars(m.cfg)
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": m.state_dict(), "config": cfg_dict, "step": 1, "history": []}, path)
    loaded, ckpt = load_checkpoint(str(path))
    assert loaded.num_params() == m.num_params()
    assert ckpt["step"] == 1
    # weights actually restored
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), loaded.named_parameters()):
        assert torch.equal(p1, p2)
