import numpy as np
import pytest

from nueronce.engine.segment import segment_ids_from_boundaries


def test_strict_capacity_raises_instead_of_silently_clipping():
    boundary_prob = np.ones((1, 16), dtype=np.float64)

    with pytest.raises(RuntimeError, match="segmentation capacity overflow"):
        segment_ids_from_boundaries(
            boundary_prob,
            min_patch=1,
            max_patch=16,
            p_max=4,
            strict_capacity=True,
        )


def test_default_capacity_behavior_remains_backward_compatible():
    boundary_prob = np.ones((1, 16), dtype=np.float64)
    segment_ids, units = segment_ids_from_boundaries(
        boundary_prob,
        min_patch=1,
        max_patch=16,
        p_max=4,
    )

    assert int(segment_ids.max()) == 3
    assert int(units[0]) == 4
