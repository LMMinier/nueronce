import torch

from nueronce.model import ModelConfig, NUERONCEModel
from nueronce.rft import (
    CanonicalRFT,
    RFTGatedMLP,
    RFTSparseLinear,
    canonical_rft_basis,
)
from nueronce.rft_model import RFTNUERONCEModel, enable_rft_ffn


def test_canonical_basis_is_unitary():
    basis = canonical_rft_basis(32, dtype=torch.complex128)
    identity = basis.conj().T @ basis
    expected = torch.eye(32, dtype=identity.dtype)
    assert torch.max(torch.abs(identity - expected)) < 1e-10


def test_analysis_synthesis_roundtrip_and_norm():
    torch.manual_seed(1)
    layer = CanonicalRFT(24, block_size=8, complex_dtype=torch.complex128)
    x = torch.randn(3, 7, 24, dtype=torch.float64)
    coefficients = layer.analysis(x)
    reconstructed = layer.synthesis(coefficients).real
    assert torch.max(torch.abs(reconstructed - x)) < 1e-10
    assert torch.allclose(
        coefficients.abs().square().sum(-1),
        x.square().sum(-1),
        atol=1e-10,
        rtol=1e-10,
    )


def test_sparse_linear_shape_gradient_and_storage():
    torch.manual_seed(2)
    layer = RFTSparseLinear(32, 48, fan_in=4, block_size=16)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 5, 48)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight_real.grad is not None
    assert layer.active_connections == 48 * 4
    assert layer.active_connections < layer.dense_connections


def test_rft_gated_mlp_trains_one_step():
    torch.manual_seed(3)
    layer = RFTGatedMLP(32, 64, fan_in=4, block_size=16)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2)
    x = torch.randn(4, 6, 32)
    target = torch.randn_like(x)
    before = (layer(x) - target).square().mean()
    for _ in range(4):
        optimizer.zero_grad()
        loss = (layer(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    after = (layer(x) - target).square().mean()
    assert after < before


def test_sparse_ffn_uses_fewer_trainable_scalars_than_dense():
    dim, hidden, fan_in = 32, 64, 2
    layer = RFTGatedMLP(dim, hidden, fan_in=fan_in, block_size=16)
    sparse = sum(parameter.numel() for parameter in layer.parameters())
    dense = 2 * dim * hidden + hidden * dim
    assert sparse < dense * 0.15


def test_existing_model_can_be_upgraded_without_changing_default_model():
    cfg = ModelConfig(
        byte_embed_dim=16,
        d_local=24,
        d_model=32,
        p_max=8,
        physical_blocks=1,
        logical_depth=1,
        n_heads=4,
        unit_window=8,
        decoder_window=8,
        decoder_layers=1,
        d_state=4,
        channel_dim=4,
        ret_byte_dim=8,
    )
    baseline = NUERONCEModel(cfg)
    assert not isinstance(baseline.core.blocks[0].ffn, RFTGatedMLP)

    upgraded = enable_rft_ffn(
        baseline,
        fan_in=2,
        block_size=16,
        ffn_mult=2,
    )
    assert isinstance(upgraded.core.blocks[0].ffn, RFTGatedMLP)

    ids = torch.randint(0, 256, (1, 24))
    loss, stats = upgraded.loss(ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["bpb"] > 0


def test_rft_nueronce_model_reports_spectral_density():
    cfg = ModelConfig(
        byte_embed_dim=16,
        d_local=24,
        d_model=32,
        p_max=8,
        physical_blocks=1,
        logical_depth=1,
        n_heads=4,
        unit_window=8,
        decoder_window=8,
        decoder_layers=1,
        d_state=4,
        channel_dim=4,
        ret_byte_dim=8,
    )
    model = RFTNUERONCEModel(
        cfg,
        rft_fan_in=2,
        rft_block_size=16,
        rft_ffn_mult=2,
    )
    stats = model.rft_stats()
    assert stats["config"]["enabled"] is True
    assert stats["layers"][0]["up_density"] == 2 / 32
    assert stats["total_rft_parameters"] > 0
