import numpy as np

from nueronce.memory import consolidation_decision, consolidation_score
from nueronce.retrieval import combine_scores
from nueronce.runtime import LoRAAdapter
from nueronce.schemas import RECORDS, load_example, load_schema


def test_combine_scores_weighting():
    s = combine_scores(
        dense_score=1.0, sparse_score=10.0, late_score=10.0,
        temporal_validity=1.0, provenance_quality=1.0, contradiction_penalty=0.0,
    )
    # dense(0.30*1) + sparse(0.20*~0.909) + late(0.25*~0.909) + temporal(0.10) + prov(0.10)
    assert 0.8 < s < 1.0
    # Contradiction strictly lowers the score.
    s2 = combine_scores(1.0, 10.0, 10.0, 1.0, 1.0, contradiction_penalty=1.0)
    assert s2 < s


def test_consolidation_score_and_decision():
    high = consolidation_score(
        corroboration=1.0, source_diversity=1.0, authority=1.0,
        temporal_stability=1.0, contradiction=0.0,
    )
    assert consolidation_decision(high, contradiction=0.0) == "semantic"
    # High contradiction blocks promotion even with a high score.
    assert consolidation_decision(high, contradiction=0.9) == "review_queue"
    low = consolidation_score(0.1, 0.1, 0.1, 0.1, contradiction=0.9)
    assert consolidation_decision(low, contradiction=0.9) == "episodic_only"


def test_lora_starts_as_zero_delta_then_adapts():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((5, 3))
    x = rng.standard_normal((2, 3))
    ad = LoRAAdapter(d_in=3, d_out=5)
    # B is zero-initialized, so the initial delta is exactly zero.
    base = x @ W.T
    assert np.allclose(ad.apply(x, W), base)
    # After perturbing B, the output diverges from the base projection.
    ad.B = rng.standard_normal(ad.B.shape)
    assert not np.allclose(ad.apply(x, W), base)


def test_schema_examples_match_required_fields():
    for name in RECORDS:
        schema = load_schema(name)
        example = load_example(name)
        for req in schema.get("required", []):
            assert req in example, f"{name} example missing required field {req!r}"
