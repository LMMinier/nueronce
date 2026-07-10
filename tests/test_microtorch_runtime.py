import numpy as np

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from cfna.microtorch.runtime import (
    BlockStateManager,
    BlockStreamFactor,
    DecomposedTrainer,
    ExecutionPlan,
)


def tiny_model():
    return MicroCFNAModel(
        MicroModelConfig(
            byte_embed_dim=8,
            d_local=12,
            d_model=16,
            p_max=8,
            physical_blocks=1,
            logical_depth=1,
            n_heads=2,
            unit_window=8,
            decoder_window=8,
            decoder_layers=1,
            d_state=4,
            channel_dim=4,
            ret_byte_dim=4,
            min_patch=2,
            max_patch=8,
        )
    )


def test_plan_covers_model_parameters(tmp_path):
    model = tiny_model()
    plan = ExecutionPlan.from_cfna(model)
    plan.validate(model)
    assert sum(b.parameter_count for b in plan.blocks) == model.num_params()
    BlockStateManager(str(tmp_path)).manifest(plan)
    assert (tmp_path / "manifest.json").exists()


def test_decomposed_training_updates_and_resumes_state(tmp_path):
    np.random.seed(3)
    model = tiny_model()
    plan = ExecutionPlan.from_cfna(model)
    trainer = DecomposedTrainer(
        model,
        plan,
        BlockStateManager(str(tmp_path)),
        BlockStreamFactor(lr=2e-3, tile_rows=16),
    )
    batch = np.array([[ord(c) for c in "logic: 2+3=5\n"]])
    before = [p.data.copy() for p in model.parameters()]
    first = trainer.train_step(lambda: model.lm_loss(batch))
    second = trainer.train_step(lambda: model.lm_loss(batch))
    assert np.isfinite(first["loss"]) and np.isfinite(second["loss"])
    assert any(
        not np.array_equal(a, p.data) for a, p in zip(before, model.parameters())
    )
    assert all((tmp_path / f"{b.name}.pkl").exists() for b in plan.blocks)
