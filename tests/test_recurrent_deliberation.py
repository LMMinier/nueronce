"""Falsifiable invariants for convergent recurrent deliberation."""

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from nueronce.blocks import HybridCoreStack
from nueronce.engine.nueronce_blocks import HybridCoreStack as EngineCore
from nueronce.engine.tensor import Tensor as EngineTensor


def test_torch_equilibrium_adds_no_parameters_and_halts():
    torch.manual_seed(1)
    core = HybridCoreStack(16, physical_blocks=1, n_heads=2,
                           local_window=4, sparse_topk=2, d_state=4)
    count = sum(p.numel() for p in core.parameters())
    x = torch.randn(1, 6, 16)
    out = core(x, 8, reasoning_mode="equilibrium", min_depth=2,
               halt_epsilon=1e9)
    assert out.shape == x.shape
    assert sum(p.numel() for p in core.parameters()) == count
    assert core.last_reasoning_stats["steps"] == 2
    assert core.last_reasoning_stats["halted"] is True


def test_torch_fixed_mode_is_exact_legacy_recurrence():
    torch.manual_seed(2)
    core = HybridCoreStack(16, physical_blocks=1, n_heads=2,
                           local_window=4, sparse_topk=2, d_state=4)
    x = torch.randn(1, 6, 16)
    expected = x
    for _ in range(3):
        expected = core.blocks[0](expected)
    actual = core(x, 3, reasoning_mode="fixed")
    torch.testing.assert_close(actual, expected)


def test_collect_states_exposes_each_differentiable_transition():
    torch.manual_seed(3)
    core = HybridCoreStack(16, physical_blocks=1, n_heads=2,
                           local_window=4, sparse_topk=2, d_state=4)
    x = torch.randn(1, 6, 16, requires_grad=True)
    final, states = core(x, 3, reasoning_mode="equilibrium", collect_states=True)
    assert len(states) == 3
    torch.testing.assert_close(final, states[-1])
    sum(state.mean() for state in states).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_engine_equilibrium_adds_no_parameters_and_halts():
    np.random.seed(1)
    core = EngineCore(16, physical_blocks=1, n_heads=2,
                      local_window=4, sparse_topk=2, d_state=4)
    count = sum(p.data.size for p in core.parameters())
    x = EngineTensor(np.random.randn(1, 6, 16))
    out = core(x, 8, reasoning_mode="equilibrium", min_depth=2,
               halt_epsilon=1e9)
    assert out.shape == x.shape
    assert sum(p.data.size for p in core.parameters()) == count
    assert core.last_reasoning_stats["steps"] == 2
    assert core.last_reasoning_stats["halted"] is True
