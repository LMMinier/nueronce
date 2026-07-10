"""The H2 channel-ablation mechanism (Lane G probe plumbing).

Asserts the ablation hook itself is correct — not the H2 verdict, which is a
measured empirical outcome recorded in benchmarks/h2_channel_probe.json. No
torch needed.
"""

import numpy as np

from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.engine.tensor import no_grad
from nueronce.types import CHANNELS


def _tiny():
    return NueronceModel(NueronceConfig(
        byte_embed_dim=12, d_local=16, d_model=24, p_max=12, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=8, decoder_window=12,
        decoder_layers=1, d_state=6, channel_dim=8, min_patch=2, max_patch=10))


def test_channel_mask_none_is_a_noop():
    m = _tiny()
    ids = np.random.randint(0, 256, size=(2, 24))
    with no_grad():
        a = m.forward(ids)[0].data
        m.memory._probe_channel_mask = np.ones(len(CHANNELS))
        b = m.forward(ids)[0].data
        m.memory._probe_channel_mask = None
    assert np.allclose(a, b)  # all-ones mask == unmasked


def test_zeroing_all_channels_changes_output_more_than_one():
    m = _tiny()
    ids = np.random.randint(0, 256, size=(2, 24))
    with no_grad():
        base = m.forward(ids)[0].data
        one = np.ones(len(CHANNELS)); one[0] = 0.0
        m.memory._probe_channel_mask = one
        d_one = np.abs(m.forward(ids)[0].data - base).mean()
        m.memory._probe_channel_mask = np.zeros(len(CHANNELS))
        d_all = np.abs(m.forward(ids)[0].data - base).mean()
        m.memory._probe_channel_mask = None
    assert d_all >= d_one  # removing all channels perturbs at least as much as one
    assert d_all > 0        # the hook actually does something
