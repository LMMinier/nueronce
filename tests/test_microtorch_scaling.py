"""The 350M budget is a real, constructable, counted model on microtorch too
— not just on PyTorch. Mirrors tests/test_scaling.py. No PyTorch needed."""

import numpy as np

from cfna.config import DEFAULT_CONFIG
from cfna.microtorch.cfna_model import MicroCFNAModel, large_config
from cfna.microtorch.tensor import no_grad


def test_large_config_builds_at_target_scale():
    m = MicroCFNAModel(large_config())
    n = m.num_params()
    # ~337M in practice; assert it is genuinely at the hundreds-of-millions scale.
    assert 250_000_000 < n < 450_000_000, f"unexpected param count: {n:,}"
    # and within ~25% of the design's documented 350M budget.
    budget = DEFAULT_CONFIG.total_param_budget_m * 1e6
    assert abs(n - budget) / budget < 0.25


def test_large_config_forward_on_tiny_input():
    # one tiny forward to prove the big model is wired, not just allocated.
    m = MicroCFNAModel(large_config())
    ids = np.random.randint(0, 256, size=(1, 16))
    with no_grad():
        logits, _ = m.forward(ids)
    assert logits.shape == (1, 16, 256)
