"""Acceptance tests for Cognitive Suite V2: determinism, module necessity,
poison resistance, and baseline superiority on value accuracy.

Kept small (few seeds / trials) so the suite stays fast; the full 5-seed x 1050
run lives in scripts/eval_cognitive_v2.py.
"""

from __future__ import annotations

import random

from nueronce.cognition_v2 import (FAMILIES, STRATEGIES, generate_suite, is_correct,
                               policy)
from nueronce.cognitive_eval import run
from nueronce.cognitive_holdouts import generate_holdouts


def _acc(trials, strategy):
    rng = random.Random(0)
    return sum(is_correct(t, STRATEGIES[strategy](t, rng)) for t in trials) / len(trials)


def test_generator_is_deterministic():
    a = generate_suite(3, 90)
    b = generate_suite(3, 90)
    assert [repr(t) for t in a] == [repr(t) for t in b]
    assert generate_suite(4, 90) != generate_suite(3, 90)


def test_all_families_present():
    trials = generate_suite(1, len(FAMILIES) * 3)
    assert set(t.family for t in trials) == set(FAMILIES)


def test_full_policy_is_correct_on_its_spec():
    trials = generate_suite(7, 300)
    assert _acc(trials, "FULL_COGNITIVE_LOOP") == 1.0


def test_full_policy_never_poisoned():
    trials = generate_suite(7, 300)
    rng = random.Random(0)
    for t in trials:
        if t.poison_value is not None:
            v = policy(t)
            assert v.answer_value != t.poison_value


def test_every_ablation_is_worse_than_full():
    trials = generate_suite(11, 300)
    full = _acc(trials, "FULL_COGNITIVE_LOOP")
    for abl in ("NO_AUTHORITY", "NO_SUPERSESSION", "NO_RETRIEVAL", "NO_PLANNING", "NO_VERIFICATION"):
        assert _acc(trials, abl) < full, abl


def test_full_beats_baselines_on_value_accuracy():
    trials = generate_suite(5, 300)
    from nueronce.cognition_v2 import Verdict  # noqa

    def value_acc(strategy):
        rng = random.Random(0)
        vt = [(t, STRATEGIES[strategy](t, rng)) for t in trials]
        vt = [(t, v) for t, v in vt if t.gold_value is not None and not t.gold_conflict and not t.gold_uncertain]
        return sum(v.answer_value == t.gold_value for t, v in vt) / len(vt)

    full = value_acc("FULL_COGNITIVE_LOOP")
    assert full - value_acc("NEWEST_FACT_WINS") >= 0.10
    assert full - value_acc("KEYWORD_RULE_ENGINE") >= 0.10


def test_holdouts_resisted():
    hold = generate_holdouts(2, 160)
    assert _acc(hold, "FULL_COGNITIVE_LOOP") >= 0.85


def test_gate_passes_small_scale():
    res = run([1, 2], 150, 80)
    assert res["gate"]["passed"], res["gate"]["checks"]
