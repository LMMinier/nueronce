"""Chat interface wiring on engine (tiny untrained model — output quality
not asserted), mirroring tests/test_chat.py. No PyTorch needed."""

import numpy as np

from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.engine.chat import MicroConversation, load_checkpoint
from nueronce.training.sharded_sft import save_checkpoint


def _tiny():
    return NueronceModel(NueronceConfig(
        byte_embed_dim=16, d_local=24, d_model=32, p_max=16, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8, min_patch=2, max_patch=14))


def test_conversation_turn_returns_text_and_tracks_transcript():
    np.random.seed(0)
    convo = MicroConversation(_tiny(), temperature=0.7, max_new=12, min_new=2)
    reply = convo.say("Hello there, who are you?")
    assert isinstance(reply, str)
    assert len(convo.transcript) == 2            # user + assistant recorded
    assert convo.transcript[0] == ("user", "Hello there, who are you?")


def test_checkpoint_roundtrip(tmp_path):
    from nueronce.engine.optim import AdamW

    m = _tiny()
    opt = AdamW(list(m.parameters()), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), m, opt, {"step": 1, "history": []})

    loaded, ckpt = load_checkpoint(str(path))
    assert loaded.num_params() == m.num_params()
    assert ckpt["meta"]["step"] == 1
    # weights actually restored
    for p1, p2 in zip(m.parameters(), loaded.parameters()):
        assert np.array_equal(p1.data, p2.data)
