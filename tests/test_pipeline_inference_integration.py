import pytest

torch = pytest.importorskip("torch")

from nueronce.pipeline import ModelRenderer, evidence_to_retrieval_tensors, verification_feedback
from nueronce.verification import IndependentVerifier
from nueronce.impl import default_verifier_hooks
from nueronce.types import MemoryRecord, VerificationFailure, VerificationReport


def _memory(mid: str, content: str, status: str = "verified") -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid,
        memory_type="semantic",
        content=content,
        source_ids=[mid],
        embeddings={},
        structured_repr={},
        authority_level="verified_secondary_source" if status == "verified" else "unverified_external_content",
        creation_time="t",
        last_verified_time="t",
        confidence=0.9,
        authenticity_status=status,
    )


class CaptureModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt.decode("utf-8"), kwargs))
        return b"Grounded answer.<|end|>"


def test_renderer_prompt_contains_query_evidence_plan_reasoning_and_tool_outputs():
    model = CaptureModel()
    evidence = [_memory("doc1", "Liberty requires accountable law.")]
    tool = [_memory("tool1", "calculator result: 42")]
    neighbor_ids, neighbor_mask = evidence_to_retrieval_tensors(evidence)
    renderer = ModelRenderer(model)
    text = renderer.render({
        "query": "What is liberty?",
        "evidence": evidence,
        "reasoning": {"best_hypothesis": "answer from doc1"},
        "plan": {"section_order": ["answer"], "central_answer": "cite doc1"},
        "tool_outputs": tool,
        "neighbor_ids": neighbor_ids,
        "neighbor_mask": neighbor_mask,
    }, {})
    prompt, kwargs = model.calls[0]
    assert text == "Grounded answer."
    assert "What is liberty?" in prompt
    assert "Liberty requires accountable law." in prompt
    assert "answer from doc1" in prompt
    assert "cite doc1" in prompt
    assert "calculator result: 42" in prompt
    assert kwargs["neighbor_ids"] is neighbor_ids
    assert kwargs["neighbor_mask"] is neighbor_mask
    assert kwargs["continuation_only"] is True


def test_renderer_revision_prompt_includes_verifier_feedback_and_reuses_evidence():
    model = CaptureModel()
    evidence = [_memory("doc1", "Only this evidence is trusted.")]
    neighbor_ids, neighbor_mask = evidence_to_retrieval_tensors(evidence)
    renderer = ModelRenderer(model)
    out = renderer.render_revision({
        "query": "Answer carefully",
        "evidence": evidence,
        "reasoning": {},
        "plan": {"central_answer": "Use doc1"},
        "neighbor_ids": neighbor_ids,
        "neighbor_mask": neighbor_mask,
    }, "Unsupported first draft.", {"unsupported_claims": ["Unsupported first draft."], "passed": False})
    prompt, kwargs = model.calls[0]
    assert out == "Grounded answer."
    assert "Only this evidence is trusted." in prompt
    assert "Unsupported first draft." in prompt
    assert "unsupported_claims" in prompt
    assert kwargs["neighbor_ids"] is neighbor_ids


def test_verification_feedback_is_structured():
    report = VerificationReport(
        passes=False,
        failures=[
            VerificationFailure("unsupported_claim", "bad claim", 1.0, "remove bad claim"),
            VerificationFailure("contradiction", "conflict", 1.0, "resolve conflict"),
            VerificationFailure("miscalibration", None, 0.7, "hedge"),
        ],
        supported_claim_fraction=0.0,
        contradiction_fraction=1.0,
        calibration_error=1.0,
    )
    feedback = verification_feedback(report)
    assert feedback["passed"] is False
    assert feedback["unsupported_claims"] == ["bad claim"]
    assert feedback["contradictions"] == ["conflict"]
    assert "hedge" in feedback["format_failures"]


def test_surface_quality_failures_block_verification():
    verifier = IndependentVerifier(default_verifier_hooks())
    report = verifier.verify(
        "the the the the the the",
        {"user_goal": "What does NUERONCE separate?", "completion_checklist": []},
        [_memory("doc1", "NUERONCE separates retrieval reasoning and generation.")],
        [],
    )
    assert not report.passes
    assert any(f.category == "surface_quality" for f in report.failures)


def test_repetitive_first_draft_can_trigger_one_revision_then_fail_if_still_bad():
    class LoopModel(CaptureModel):
        def generate(self, prompt, **kwargs):
            self.calls.append((prompt.decode("utf-8"), kwargs))
            return b"the the the the the the"

    model = LoopModel()
    evidence = [_memory("doc1", "NUERONCE separates retrieval reasoning and generation.")]
    neighbor_ids, neighbor_mask = evidence_to_retrieval_tensors(evidence)
    renderer = ModelRenderer(model)
    draft = {
        "query": "What does NUERONCE separate?",
        "evidence": evidence,
        "reasoning": {},
        "plan": {"user_goal": "What does NUERONCE separate?", "completion_checklist": []},
        "neighbor_ids": neighbor_ids,
        "neighbor_mask": neighbor_mask,
    }
    verifier = IndependentVerifier(default_verifier_hooks())
    first = renderer.render(draft, {})
    first_report = verifier.verify(first, draft["plan"], evidence, [])
    second = renderer.render_revision(draft, first, verification_feedback(first_report))
    second_report = verifier.verify(second, draft["plan"], evidence, [])
    assert not first_report.passes
    assert not second_report.passes
    assert len(model.calls) == 2
