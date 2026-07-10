"""V3.3 blind-style multi-document provenance benchmark tests."""

from __future__ import annotations

from nueronce.provenance_v33 import (
    ABLATIONS,
    SYSTEMS,
    dev_cases_json,
    generate_dev_cases,
    public_case,
    run,
)


def test_v33_generates_at_least_100_multi_document_dev_cases():
    cases = generate_dev_cases(seed=0, n=100)

    assert len(cases) == 100
    assert all(3 <= len(c.documents) <= 8 for c in cases)
    assert len({c.family for c in cases}) >= 10


def test_public_case_does_not_expose_hidden_gold():
    case = generate_dev_cases(seed=1, n=1)[0]
    pub = public_case(case)

    assert "hidden_gold" not in pub
    assert "documents" in pub
    assert all("expected_authenticity" not in d for d in pub["documents"])


def test_dev_artifact_marks_final_blind_labels_withheld():
    dev = dev_cases_json(seed=0, n=5)

    assert dev["split"] == "development"
    assert "withheld" in dev["final_blind_labels"]
    assert len(dev["cases"]) == 5


def test_v33_reports_required_systems_metrics_and_ablations():
    res = run(seed=0, n=24)

    assert set(res["systems"]) == set(SYSTEMS)
    assert set(res["ablations"]) == set(ABLATIONS)
    for system in SYSTEMS:
        metrics = res["systems"][system]["metrics"]
        for key in (
            "answer_accuracy",
            "source_selection_precision",
            "source_selection_recall",
            "citation_precision",
            "citation_recall",
            "unsupported_claim_rate",
            "poison_acceptance",
            "false_rejection",
            "abstention_rate",
            "coverage",
            "selective_accuracy",
            "safe_outcome_rate",
            "conflict_detection_accuracy",
            "supersession_accuracy",
            "temporal_accuracy",
            "scope_accuracy",
            "utility",
            "mean_latency_ms",
            "mean_peak_memory_kb",
        ):
            assert key in metrics


def test_v33_expected_mechanisms_have_interpretable_effects():
    res = run(seed=0, n=36)
    full = res["ablations"]["full_pipeline"]["metrics"]

    assert full["poison_acceptance"] <= res["ablations"]["minus_provenance"]["metrics"]["poison_acceptance"]
    assert full["unsupported_claim_rate"] < res["ablations"]["minus_verifier"]["metrics"]["unsupported_claim_rate"]
    assert full["safe_outcome_rate"] >= res["ablations"]["minus_contract"]["metrics"]["safe_outcome_rate"]
    assert "full_beats_contract_only" in res["gate"]
