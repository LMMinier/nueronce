"""Milestone tests for the V3.1 provenance-gate benchmark."""

from __future__ import annotations

from nueronce.provenance_bench import run


def test_provenance_gate_reduces_poison_without_excessive_rejection():
    res = run(seed=0, n=160)
    classifier_only = res["classifier_only"]
    gated = res["gated"]

    assert classifier_only["poison_acceptance"] > 0.50
    assert gated["poison_acceptance"] < 0.05
    assert gated["poison_acceptance"] < classifier_only["poison_acceptance"]
    assert gated["abstain_or_escalate"] <= 0.25
    assert gated["accuracy"] >= 0.75


def test_compromised_key_requires_revocation():
    res = run(seed=1, n=40)
    compromised = res["compromised_key"]

    assert compromised["stolen_key_before_revocation"] == "verified"
    assert compromised["after_revocation"] == "revoked"
