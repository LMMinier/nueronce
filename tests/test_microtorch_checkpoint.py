import copy

import numpy as np

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig


def tiny_cfg(checkpointing=False):
    return MicroModelConfig(
        byte_embed_dim=8, d_local=12, d_model=16, p_max=8,
        physical_blocks=1, logical_depth=1, n_heads=2,
        unit_window=8, decoder_window=8, decoder_layers=1,
        d_state=4, channel_dim=4, ret_byte_dim=4,
        min_patch=2, max_patch=8,
        activation_checkpointing=checkpointing,
    )


def graph_nodes(tensor):
    seen, stack = set(), [tensor]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node._prev)
    return len(seen)


def matched_models():
    np.random.seed(11)
    baseline = MicroCFNAModel(tiny_cfg(False))
    replay = MicroCFNAModel(tiny_cfg(True))
    for source, target in zip(baseline.parameters(), replay.parameters()):
        target.data = source.data.copy()
    return baseline, replay


def test_activation_checkpointing_matches_loss_and_all_gradients():
    baseline, replay = matched_models()
    ids = np.array([[108, 111, 103, 105, 99, 58, 32, 50, 43, 51, 61, 53, 10]])
    loss_a = baseline.lm_loss(ids)
    loss_b = replay.lm_loss(ids)
    loss_a.backward()
    loss_b.backward()
    assert loss_a.item() == loss_b.item()
    for pa, pb in zip(baseline.parameters(), replay.parameters()):
        if pa.grad is None or pb.grad is None:
            assert pa.grad is pb.grad
        else:
            np.testing.assert_allclose(pa.grad, pb.grad, rtol=1e-9, atol=1e-10)


def test_activation_checkpointing_reduces_resident_graph_nodes():
    baseline, replay = matched_models()
    ids = np.array([[108, 111, 103, 105, 99, 58, 32, 50, 43, 51, 61, 53, 10]])
    global_nodes = graph_nodes(baseline.lm_loss(ids))
    checkpoint_nodes = graph_nodes(replay.lm_loss(ids))
    assert checkpoint_nodes < global_nodes * 0.5
