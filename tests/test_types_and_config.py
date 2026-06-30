"""Data-model and config sanity checks."""

from cfna import config, types


def test_authority_order_matches_literal():
    # Every ordered authority level is a recognized AuthorityLevel value.
    assert types.AUTHORITY_ORDER[0] == "system_policy"
    assert types.AUTHORITY_ORDER[-1] == "generated_hypothesis"
    assert len(types.AUTHORITY_ORDER) == 8


def test_channels_constants():
    assert types.K_CHANNELS == len(types.CHANNELS) == 7
    assert "evid" in types.CHANNELS and "auth" in types.CHANNELS


def test_dataclass_defaults_construct():
    ku = types.KnowledgeUnit(
        unit_id="u1", source_id="s1", unit_type="claim",
        text="x", byte_span=(0, 1), section_path=["1"],
    )
    assert ku.concepts == [] and ku.confidence_target == 0.5

    mr = types.MemoryRecord(
        memory_id="m1", memory_type="episodic", content="c",
        source_ids=[], embeddings={}, structured_repr={},
        authority_level="tool_observation", creation_time="t",
        last_verified_time=None, confidence=0.9,
    )
    assert mr.privacy_scope == "session"
    assert mr.consolidation_status == "episodic_only"


def test_default_config_param_budget():
    cfg = config.DEFAULT_CONFIG
    # The documented prototype targets ~350M parameters.
    assert abs(cfg.total_param_budget_m - 350.0) < 1e-6
    assert cfg.core.d_model == 1536
    assert cfg.core.logical_depth["RESEARCH"] == 24
    assert set(cfg.memory.retention) == set(types.CHANNELS)


def test_retrieval_weights_present():
    r = config.DEFAULT_CONFIG.retrieval
    # Matches the design doc fusion: additive weights sum to 0.95, with the
    # contradiction penalty (0.05) subtracted so all magnitudes sum to 1.0.
    additive = r.w_dense + r.w_sparse + r.w_late + r.w_temporal + r.w_provenance
    assert abs(additive - 0.95) < 1e-9
    assert abs(additive + r.w_contradiction - 1.0) < 1e-9
