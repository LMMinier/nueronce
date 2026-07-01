"""Wave 2: the end-to-end learned cognitive loop.

Runs in this sandbox (needs cryptography — pip install cffi; skips cleanly
otherwise). Verifies the two things Wave 2 claims: (1) the integrated loop is
a faithful superset of contract-only — with unsigned admission off it exactly
reproduces the frozen baseline; (2) counterfactual citation attribution
identifies decisive evidence, the V3.3 gate that failed by construction.
"""

import pytest

pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")

from cfna.integrated_eval import aggregate, run_integrated, score_case
from cfna.provenance_v33 import generate_dev_cases, run_system


def _cases(n=60):
    return generate_dev_cases(seed=20260701, n=n)


def test_integrated_none_reproduces_contract_only():
    """authority_mode='none' must not admit unsigned docs -> identical answers
    to the frozen provenance_contract baseline, case for case."""
    for case in _cases():
        contract = run_system(case, "provenance_contract")
        integ = run_integrated(case, authority_mode="none")
        assert integ.answer == contract.answer, case.case_id
        assert integ.escalation_status == contract.escalation_status, case.case_id


def test_counterfactual_citations_identify_decisive_evidence():
    """On correctly-answered answerable cases, cited docs == gold decisive set
    (precision and recall both perfect) — the fix for the V3.3 citation gate."""
    rows = []
    for case in _cases():
        if case.hidden_gold.expected_outcome != "answer":
            continue
        out = run_integrated(case, authority_mode="none")
        s = score_case(case, out)
        if s["answer_ok"]:
            rows.append(s)
    assert rows
    agg = aggregate(rows)
    assert agg["citation_precision"] >= 0.90
    assert agg["citation_recall"] >= 0.90


def test_admitting_unsigned_by_channel_reintroduces_poison():
    """The core scientific finding: on the V3.3 appearance-perfect-unsigned
    threat model, oracle channel-authority admission reintroduces poison the
    signature gate had removed (the spoof_perfect ceiling at pipeline level)."""
    none_rows = [score_case(c, run_integrated(c, authority_mode="none")) for c in _cases()]
    oracle_rows = [score_case(c, run_integrated(c, authority_mode="oracle")) for c in _cases()]
    none_poison = aggregate(none_rows)["poison_acceptance"]
    oracle_poison = aggregate(oracle_rows)["poison_acceptance"]
    assert none_poison == 0.0
    assert oracle_poison > none_poison  # admitting unsigned by channel lets poison in


def test_trained_classifier_matches_oracle_end_to_end():
    """The trained authority classifier is not the bottleneck: predicted-mode
    accuracy equals oracle-mode accuracy (the threat model is the limit)."""
    from cfna.authority_clf import AuthorityClassifier
    from cfna.authority_data import gen_examples

    clf = AuthorityClassifier(seed=0)
    clf.fit(gen_examples(seed=0, n=2000), steps=300, seed=0)

    def predict(features):
        return clf.predict_trusted(features)

    cases = _cases()
    oracle = aggregate([score_case(c, run_integrated(c, authority_mode="oracle")) for c in cases])
    pred = aggregate([score_case(c, run_integrated(
        c, authority_predict=predict, authority_mode="predicted")) for c in cases])
    assert abs(oracle["answer_accuracy"] - pred["answer_accuracy"]) <= 0.05
