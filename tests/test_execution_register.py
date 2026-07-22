import pytest

torch = pytest.importorskip("torch")

from nueronce.reasoning import AddressableExecutionRegister
from nueronce.engine.reasoning import AddressableExecutionRegister as EngineExecutor
from nueronce.engine.tensor import Tensor as EngineTensor
import numpy as np


def test_execution_register_shapes_gradients_and_shared_parameters():
    torch.manual_seed(0)
    machine = AddressableExecutionRegister(16)
    state = torch.randn(3, 16, requires_grad=True)
    keys = torch.randn(3, 5, 16)
    values = torch.randn(3, 5, 16)
    params = sum(p.numel() for p in machine.parameters())
    final, trace = machine(state, keys, values, 4)
    assert final.shape == state.shape
    assert len(trace.states) == len(trace.read_weights) == len(trace.halt_probabilities) == 4
    assert trace.read_weights[0].shape == (3, 5)
    assert sum(p.numel() for p in machine.parameters()) == params
    final.sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_execution_register_halt_is_batch_safe():
    machine = AddressableExecutionRegister(8)
    state = torch.randn(2, 8)
    keys = values = torch.randn(2, 3, 8)
    _, trace = machine(state, keys, values, 6, halt_threshold=0.0, min_steps=2)
    assert len(trace.states) == 6


def test_causal_multiquery_execution_and_engine_shape():
    torch.manual_seed(2)
    machine = AddressableExecutionRegister(8)
    state = keys = values = torch.randn(2, 4, 8)
    mask = torch.ones(4, 4, dtype=torch.bool).tril()[None].expand(2, -1, -1)
    final, trace = machine(state, keys, values, 2, memory_mask=mask)
    assert final.shape == (2, 4, 8)
    assert trace.read_weights[0].shape == (2, 4, 4)

    engine = EngineExecutor(8)
    array = np.random.randn(2, 4, 8)
    e_final, e_trace = engine(EngineTensor(array), EngineTensor(array),
                              EngineTensor(array), 2, memory_mask=mask.numpy())
    assert e_final.shape == (2, 4, 8)
    assert e_trace.read_weights[0].shape == (2, 4, 4)
