from __future__ import annotations

import math

import torch

from cfna.model import CFNAModel, ModelConfig
from cfna.training.cpu_streaming import CPUStreamingConfig, CPUStreamingTrainer, STAGES


def tiny_model() -> CFNAModel:
    return CFNAModel(ModelConfig(
        byte_embed_dim=8, d_local=16, d_model=16, p_max=8,
        physical_blocks=1, logical_depth=1, n_heads=2,
        unit_window=4, decoder_window=4, decoder_layers=1,
        d_state=4, channel_dim=4, ret_byte_dim=4,
        min_patch=2, max_patch=6,
    ))


def test_perception_stage_updates_only_perception_subsystem():
    model = tiny_model()
    trainer = CPUStreamingTrainer(model, CPUStreamingConfig(
        chunk_size=32, steps_per_stage=10, batch_size=1,
    ))
    before_perception = [p.detach().clone() for p in model.perception.parameters()]
    before_core = [p.detach().clone() for p in model.core.parameters()]

    stats = trainer.train_step([b"bounded memory training works locally"])

    assert stats["stage"] == "perception"
    assert any(not torch.equal(a, b) for a, b in zip(before_perception, model.perception.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before_core, model.core.parameters()))
    assert trainer.optimizer is not None
    assert trainer.optimizer.state == {}


def test_all_local_stages_run_and_keep_losses_finite():
    trainer = CPUStreamingTrainer(tiny_model(), CPUStreamingConfig(
        chunk_size=32, steps_per_stage=1, batch_size=1,
    ))
    seen = []
    for _ in range(8):
        stats = trainer.train_step([b"the model learns from a short byte window"])
        seen.append(stats["stage"])
        assert math.isfinite(stats["loss"])
        assert stats["bytes"] <= 32
    assert set(seen) == set(STAGES)


def test_memory_contract_counts_only_active_parameters():
    trainer = CPUStreamingTrainer(tiny_model(), CPUStreamingConfig(chunk_size=24))
    contract = trainer.memory_contract()
    assert contract["device"] == "cpu"
    assert contract["optimizer_state_bytes"] == 0
    assert 0 < contract["active_parameters"] < contract["total_model_parameters"]
    assert contract["estimated_gradient_bytes_fp32"] == contract["active_parameters"] * 4


def test_long_inputs_are_hard_capped_to_chunk_size():
    trainer = CPUStreamingTrainer(tiny_model(), CPUStreamingConfig(chunk_size=16))
    stats = trainer.train_step([b"x" * 1000])
    assert stats["bytes"] == 16
