"""Coherent inference wrapper tests."""

from cfna.coherent_inference import (
    clean_reply,
    coherence_warning,
    deterministic_answer,
    respond,
    run_probes,
    surface_failure_reason,
)


def test_deterministic_answer_handles_arithmetic_and_known_facts():
    assert deterministic_answer("What is 17 plus 25?") == "17 plus 25 equals 42."
    assert deterministic_answer("Calculate 9 * 8.") == "9 times 8 equals 72."
    assert deterministic_answer("What is the capital of France?") == "The capital of France is Paris."


def test_clean_reply_stops_at_turn_boundary_and_sentence():
    raw = "This is a reply. User: next turn should not leak"
    assert clean_reply(raw) == "This is a reply."
    canonical = "<|assistant|>\nThis is canonical.<|end|>\n<|user|>\nnext"
    assert clean_reply(canonical) == "This is canonical."


def test_coherence_warning_catches_repetition():
    assert coherence_warning("aaaaaaaaaaaaaaaa") == "repetitive_output"
    assert coherence_warning("the the the the the the the the") == "repetitive_output"
    assert coherence_warning("A short useful answer.") is None


def test_surface_failure_reason_catches_empty_repetition_echo_and_markers():
    assert surface_failure_reason("") == "empty_output"
    assert surface_failure_reason("the the the the the the") == "repetitive_output"
    assert surface_failure_reason("What is liberty? It is freedom.", prompt="What is liberty?") == "prompt_echo"
    assert surface_failure_reason("answer <|user|> leaked") == "role_marker_leakage"


def test_respond_uses_tools_before_model_when_enabled():
    called = {"n": 0}

    def model_fn(prompt):
        called["n"] += 1
        return "wrong"

    res = respond("What is 2 plus 2?", model_fn, assist_tools=True)
    assert res.source == "tool"
    assert "4" in res.text
    assert called["n"] == 0


def test_respond_falls_back_on_bad_model_output():
    res = respond("hello", lambda _: "aaaaaaaaaaaa", assist_tools=False)
    assert res.source == "fallback"
    assert res.warning == "repetitive_output"


def test_run_probes_reports_rates():
    res = run_probes(lambda _: "I am an assistant.", assist_tools=True)
    assert res["n"] >= 3
    assert 0.0 <= res["pass_rate"] <= 1.0
    assert res["tool_rate"] > 0.0
    assert "model_only" in res
    assert "tool_assisted" in res


def test_tool_answers_do_not_count_as_model_only_success():
    res = run_probes(lambda _: "not correct.", assist_tools=True)
    assert res["tool_rate"] > 0.0
    assert res["model_only"]["exact_correctness"] == 0.0
