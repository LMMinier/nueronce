import torch

from nueronce.model import ModelConfig, NUERONCEModel
from nueronce.rft import RFTGatedMLP
from nueronce.spectral_baselines import (
    DCTSparseGatedMLP,
    SparseGatedMLP,
    orthonormal_dct_basis,
    replace_core_ffn,
)


def test_dct_basis_is_orthonormal():
    basis = orthonormal_dct_basis(32, dtype=torch.float64)
    identity = basis.T @ basis
    assert torch.max(torch.abs(identity - torch.eye(32, dtype=torch.float64))) < 1e-10


def test_sparse_variants_have_equal_trainable_scalar_budget():
    dim, hidden, fan_in = 32, 64, 4
    ordinary = SparseGatedMLP(dim, hidden, fan_in=fan_in)
    dct = DCTSparseGatedMLP(dim, hidden, fan_in=fan_in)
    rft = RFTGatedMLP(dim, hidden, fan_in=fan_in, block_size=16)
    counts = [sum(p.numel() for p in module.parameters()) for module in (ordinary, dct, rft)]
    assert counts[0] == counts[1] == counts[2]


def test_all_sparse_variants_forward_and_backward():
    torch.manual_seed(11)
    x = torch.randn(2, 5, 32, requires_grad=True)
    for module in (
        SparseGatedMLP(32, 64, fan_in=4),
        DCTSparseGatedMLP(32, 64, fan_in=4),
        RFTGatedMLP(32, 64, fan_in=4, block_size=16),
    ):
        output = module(x)
        assert output.shape == x.shape
        loss = output.square().mean()
        loss.backward(retain_graph=True)
        assert all(p.grad is not None for p in module.parameters())


def test_dense_budget_control_is_smaller_than_full_dense():
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
    full = NUERONCEModel(cfg)
    budget = NUERONCEModel(cfg)
    replace_core_ffn(budget, "dense_budget", fan_in=4, ffn_mult=2)
    full_ffn = sum(p.numel() for p in full.core.blocks[0].ffn.parameters())
    budget_ffn = sum(p.numel() for p in budget.core.blocks[0].ffn.parameters())
    assert budget_ffn < full_ffn
