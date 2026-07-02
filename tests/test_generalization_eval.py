"""The >=200-prompt generalization suite and its seen/novel classification.
No PyTorch needed."""

from cfna.microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from cfna.training.generalization_eval import (
    EvalPrompt, build_novel_prompts, classify_prompt_seen,
    run_generalization_eval, sample_memorized_probes, summarize_generalization,
)


def test_novel_prompt_suite_has_at_least_200_prompts():
    prompts = build_novel_prompts()
    assert len(prompts) >= 200
    assert all(p.intent == "novel_variation" for p in prompts)
    # a real mixture, not one category padded to 200
    assert len({p.category for p in prompts}) >= 8


def test_classify_prompt_seen_matches_normalized_training_text():
    seen = {"what is two plus two"}
    assert classify_prompt_seen("What is two plus two?", seen) is True
    assert classify_prompt_seen("What is three plus three?", seen) is False


def test_sample_memorized_probes_uses_real_training_gold_answers():
    records = [
        {"id": "a", "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
         "source": "s", "category": "greetings"},
        {"id": "b", "messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}],
         "source": "s", "category": "greetings"},
    ]
    probes = sample_memorized_probes(records, n=2, seed=0)
    assert len(probes) == 2
    assert all(p.intent == "memorized_probe" for p in probes)
    assert {p.gold for p in probes} == {"Hello!", "Goodbye!"}


def _tiny_model():
    return MicroCFNAModel(MicroModelConfig(
        byte_embed_dim=12, d_local=16, d_model=24, p_max=12, physical_blocks=1,
        logical_depth=2, n_heads=4, unit_window=8, decoder_window=12,
        decoder_layers=1, d_state=6, channel_dim=8, ret_byte_dim=8,
        min_patch=2, max_patch=10))


def test_run_generalization_eval_produces_expected_summary_shape():
    model = _tiny_model()
    prompts = [
        EvalPrompt("Hello", "greetings", "memorized_probe", gold="Hi there!"),
        EvalPrompt("What is nine plus one?", "arithmetic", "novel_variation",
                   lambda t: "10" in t),
    ]
    seen = {"hello"}
    results = run_generalization_eval(model, prompts, seen, max_new=15)
    assert results["n_total"] == 2
    assert results["memorized"]["n"] == 1
    assert results["novel"]["n"] == 1
    assert set(results["by_category"]) == {"greetings", "arithmetic"}
    assert len(results["examples"]) == 2


def test_summarize_generalization_handles_prompts_with_no_check_function():
    results = [
        {"prompt": "hi", "category": "c", "declared_intent": "novel_variation",
         "actually_seen_in_training": False, "response": "hello",
         "valid_utf8": True, "stopped_at_stop_byte": True, "coherent": True, "check_passed": None, "gold": None},
    ]
    summary = summarize_generalization(results)
    assert summary["overall"]["check_pass_rate"] is None  # no checkable prompts -> no crash, no fake rate
    assert summary["overall"]["coherent_rate"] == 1.0
