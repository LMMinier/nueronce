"""Acceptance tests for the CFNA cognitive loop and its falsification suite.

These pin the architectural claim: the full cognitive loop solves the controlled
tasks, and removing any single stage breaks exactly the tasks that need it.
"""

from __future__ import annotations

from cfna.cognition import ABLATION_FLAGS, grade, run
from cfna.cognitive_suite import default_scenarios


def _score(ablations):
    return sum(grade(sc, run(sc, ablations))[0] for sc in default_scenarios())


def test_full_loop_solves_every_scenario():
    assert _score({}) == len(default_scenarios())


def test_every_ablation_is_strictly_worse():
    full = _score({})
    for flag in ABLATION_FLAGS:
        assert _score({flag: True}) < full, f"{flag} did not degrade the suite"


def test_untrusted_document_never_overwrites_a_verified_fact():
    # The poison value must not appear in the FULL answer, and the trusted
    # correction must win with a citation.
    sc = next(s for s in default_scenarios() if s.name == "authority_overwrite")
    trace = run(sc, {})
    assert "Belport" in trace.answer
    assert "Xtown" not in trace.answer
    assert "source:" in trace.answer
    # The untrusted item is explicitly recorded as rejected, not silently dropped.
    rejected = [i.value for i in trace.reasoning.rejected_untrusted]
    assert "Xtown" in rejected


def test_removing_authority_lets_the_poison_win():
    sc = next(s for s in default_scenarios() if s.name == "authority_overwrite")
    trace = run(sc, {"no_authority": True})
    assert "Xtown" in trace.answer  # poisoned: latest assertion wins without authority


def test_removing_supersession_keeps_the_stale_fact():
    sc = next(s for s in default_scenarios() if s.name == "temporal_supersession")
    trace = run(sc, {"no_supersession": True})
    assert "Dana Lee" in trace.answer  # stuck on the older filing


def test_uncertainty_is_honest_under_full_loop():
    sc = next(s for s in default_scenarios() if s.name == "uncertainty_on_untrusted")
    trace = run(sc, {})
    assert trace.reasoning.resolved_value is None
    assert "trusted source" in trace.answer


def test_trace_logs_every_stage():
    sc = default_scenarios()[0]
    t = run(sc, {})
    for stage in (t.perception, t.semantics, t.intent, t.memory_query,
                  t.evidence, t.reasoning, t.plan, t.draft, t.verification, t.revision):
        assert stage is not None
