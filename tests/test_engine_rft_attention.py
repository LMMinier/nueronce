import numpy as np

from nueronce.engine.rft_attention import PhiRotaryMultiHeadAttention, phi_rotary
from nueronce.engine.tensor import Tensor


def test_phi_rotary_preserves_shape_and_pair_norms_with_odd_head_dim():
    rng = np.random.default_rng(7)
    x = Tensor(rng.normal(size=(2, 3, 11, 131)))
    y = phi_rotary(x)
    assert y.shape == x.shape

    pair_dim = 130
    x_pairs = x.data[..., :pair_dim].reshape(2, 3, 11, 65, 2)
    y_pairs = y.data[..., :pair_dim].reshape(2, 3, 11, 65, 2)
    np.testing.assert_allclose(
        np.sum(x_pairs * x_pairs, axis=-1),
        np.sum(y_pairs * y_pairs, axis=-1),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(x.data[..., -1], y.data[..., -1])


def test_position_zero_is_identity():
    rng = np.random.default_rng(8)
    x = Tensor(rng.normal(size=(1, 2, 5, 12)))
    y = phi_rotary(x)
    np.testing.assert_allclose(x.data[:, :, 0], y.data[:, :, 0], atol=1e-12)


def test_phi_attention_remains_causal():
    rng = np.random.default_rng(9)
    attn = PhiRotaryMultiHeadAttention(dim=16, n_heads=4, window=None)
    base = rng.normal(size=(1, 8, 16))
    changed = base.copy()
    changed[:, 6:, :] += 10.0

    y1 = attn(Tensor(base)).data
    y2 = attn(Tensor(changed)).data
    np.testing.assert_allclose(y1[:, :6], y2[:, :6], rtol=1e-10, atol=1e-10)


def test_phi_attention_adds_no_parameters():
    attn = PhiRotaryMultiHeadAttention(dim=32, n_heads=4)
    count = sum(p.data.size for p in attn.parameters())
    assert count == 4 * 32 * 32
