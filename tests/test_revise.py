"""The verify→revise loop actually revises (no more lambda d, f: d no-op)."""

import pytest

from cfna import impl
from cfna.types import MemoryRecord, VerificationFailure
from cfna.verification import IndependentVerifier, verify_and_revise


def _evidence(text: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id="e1", memory_type="semantic", content=text, source_ids=["s1"],
        embeddings={}, structured_repr={}, authority_level="verified_secondary_source",
        creation_time="t", last_verified_time="t", confidence=0.9,
    )


class _DictRenderer:
    """Renders the (revisable) draft text verbatim."""

    def render(self, draft, style):
        return draft["text"]


def test_revise_draft_removes_unsupported_claim():
    draft = {"text": "Photosynthesis converts light into sugar. Dragons breathe ice."}
    failures = [VerificationFailure("unsupported_claim", "Dragons breathe ice.", 1.0, "remove")]
    out = impl.revise_draft(draft, failures)
    assert "Dragons" not in out["text"]
    assert "Photosynthesis" in out["text"]


def test_verify_and_revise_passes_after_removing_unsupported_claim():
    evidence = [_evidence(
        "Photosynthesis converts light water and carbon dioxide into sugar and oxygen."
    )]
    verifier = IndependentVerifier(impl.default_verifier_hooks())
    draft = {"text": "Photosynthesis converts light into sugar. Dragons breathe ice."}

    text, report = verify_and_revise(
        plan={"completion_checklist": []}, semantic_draft=draft, style={},
        evidence_items=evidence, tool_obs=[], renderer=_DictRenderer(),
        verifier=verifier, revise_semantic_draft=impl.revise_draft, max_rounds=3,
    )
    # the unsupported claim was removed and the final draft now verifies
    assert "Dragons" not in text
    assert report.passes
