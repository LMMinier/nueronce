from cfna.prompting import (
    ASSISTANT,
    END,
    EVIDENCE,
    PLAN,
    USER,
    assemble_conversation_prompt,
    extract_assistant_continuation,
    format_inference_prompt,
    format_revision_prompt,
    format_training_example,
)


def test_training_and_inference_share_markers():
    prompt = format_inference_prompt(
        system_message="sys", user_request="u", trusted_evidence="ev", response_plan="plan"
    )
    full, mask = format_training_example(
        system_message="sys", user_request="u", trusted_evidence="ev",
        response_plan="plan", assistant_response="answer"
    )
    assert full.startswith(prompt.encode("utf-8"))
    assert USER in prompt and EVIDENCE in prompt and PLAN in prompt and ASSISTANT in prompt
    assert full.decode("utf-8").endswith(f"answer\n{END}\n")
    assert not any(mask[:len(prompt.encode("utf-8"))])
    assert all(mask[len(prompt.encode("utf-8")):])


def test_extract_assistant_continuation_removes_prompt_and_stop():
    text = "<|user|>\nWhat is liberty?\n<|assistant|>\nA civic ideal.\n<|end|>\n<|user|>\nNext"
    assert extract_assistant_continuation(text) == "A civic ideal."


def test_revision_prompt_contains_feedback_evidence_and_plan():
    prompt = format_revision_prompt(
        system_message="sys",
        user_request="question",
        trusted_evidence="[doc1] evidence",
        response_plan="cite doc1",
        first_draft="unsupported claim",
        verifier_feedback={"unsupported_claims": ["unsupported claim"], "passed": False},
    )
    assert "[doc1] evidence" in prompt
    assert "cite doc1" in prompt
    assert "unsupported claim" in prompt
    assert prompt.endswith(f"{ASSISTANT}\n")


def test_structured_context_truncation_preserves_current_parts():
    prompt = assemble_conversation_prompt(
        system_message="system must stay",
        current_user="current question",
        recent_turns=[("user", "old user"), ("assistant", "old assistant")],
        trusted_evidence="trusted evidence",
        response_plan="plan",
        max_chars=120,
    )
    assert "system must stay" in prompt
    assert "current question" in prompt
    assert "trusted evidence" in prompt
    assert "plan" in prompt
