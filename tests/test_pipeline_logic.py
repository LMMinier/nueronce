"""Tests for control-flow that is real even though the leaf ops are injected:
the verify→revise loop, tool authority gating, and that every module imports.
"""

import importlib

import pytest

from cfna.tools import ToolExecutor
from cfna.types import MemoryRecord, VerificationFailure
from cfna.verification import IndependentVerifier, VerifierHooks, verify_and_revise

CFNA_MODULES = [
    "cfna", "cfna.types", "cfna.ops", "cfna.config", "cfna._backend",
    "cfna.ingestion", "cfna.parsing", "cfna.perception", "cfna.embeddings",
    "cfna.memory", "cfna.routers", "cfna.retrieval", "cfna.core",
    "cfna.workspace", "cfna.planning", "cfna.verification", "cfna.tools",
    "cfna.runtime", "cfna.schemas",
    "cfna.nn", "cfna.blocks", "cfna.segment", "cfna.model", "cfna.data",
    "cfna.impl", "cfna.pipeline", "cfna.retrieval_train",
    "cfna.baselines", "cfna.eval",
    "cfna.training", "cfna.training.curriculum", "cfna.training.episodes",
    "cfna.training.losses", "cfna.training.vgrft",
]


@pytest.mark.parametrize("mod", CFNA_MODULES)
def test_module_imports(mod):
    assert importlib.import_module(mod) is not None


def _passing_hooks():
    return VerifierHooks(
        extract_claims=lambda t: [t],
        match_claim_to_evidence=lambda c, e: 1.0,
        detect_text_evidence_contradictions=lambda t, e: [],
        checklist_item_satisfied=lambda t, i: True,
        output_respects_tool_result=lambda t, o: True,
        calibration_gap=lambda t, e: 0.0,
        measure_support_fraction=lambda c, e: 1.0,
    )


class _Renderer:
    def __init__(self):
        self.calls = 0

    def render(self, draft, style):
        self.calls += 1
        return draft["text"]


def test_verify_and_revise_passes_first_round():
    verifier = IndependentVerifier(_passing_hooks())
    renderer = _Renderer()
    text, report = verify_and_revise(
        plan={"completion_checklist": []},
        semantic_draft={"text": "supported statement"},
        style={},
        evidence_items=[],
        tool_obs=[],
        renderer=renderer,
        verifier=verifier,
        revise_semantic_draft=lambda d, f: d,
        max_rounds=3,
    )
    assert report.passes
    assert renderer.calls == 1  # no revision needed


def test_verify_and_revise_revises_then_gives_up():
    # An unsupported claim is a blocking (severity 1.0) failure.
    hooks = _passing_hooks()
    hooks.match_claim_to_evidence = lambda c, e: 0.0
    verifier = IndependentVerifier(hooks)
    renderer = _Renderer()
    revisions = {"n": 0}

    def revise(draft, failures):
        revisions["n"] += 1
        return draft

    text, report = verify_and_revise(
        plan={"completion_checklist": []},
        semantic_draft={"text": "unsupported claim"},
        style={},
        evidence_items=[],
        tool_obs=[],
        renderer=renderer,
        verifier=verifier,
        revise_semantic_draft=revise,
        max_rounds=3,
    )
    assert not report.passes
    assert revisions["n"] == 3  # revised on each failing round


def test_tool_executor_authority_gate():
    ex = ToolExecutor(run_tool_safely=lambda call: {"status": "ok", "out": 1})
    with pytest.raises(PermissionError):
        ex.execute({"tool": "pytest"}, {"may_execute_tools": False})

    rec = ex.execute({"tool": "pytest"}, {"may_execute_tools": True})
    assert isinstance(rec, MemoryRecord)
    assert rec.authority_level == "tool_observation"
    assert rec.confidence == 0.98
