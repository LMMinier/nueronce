"""MCQ/QA -> SFT conversion + choice-ranking evaluation. No torch, no network."""

import numpy as np

from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.training.dataset_prep import validate_record
from nueronce.training.mcq_sft import convert_records, evaluate_mcq, normalize_mcq, normalize_qa, rank_choices

ARC_STYLE = {
    "question": "Which gas do plants absorb from the atmosphere?",
    "choices": {"text": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
                "label": ["A", "B", "C", "D"]},
    "answerKey": "B",
}
MATHQA_STYLE = {
    "Problem": "What is 6 times 7?",
    "options": "a ) 13 , b ) 42 , c ) 67 , d ) 76",
    "correct": "b",
}


def test_normalize_mcq_arc_schema():
    rec = normalize_mcq(ARC_STYLE, source="arc_easy", category="science_mcq")
    assert rec is not None
    assert validate_record(rec) is None                      # passes the pipeline validator
    assert "B) Carbon dioxide" == rec["messages"][1]["content"]
    assert "A) Oxygen" in rec["messages"][0]["content"]
    assert rec["mcq"]["answer_idx"] == 1


def test_normalize_mcq_mathqa_schema():
    rec = normalize_mcq(MATHQA_STYLE, source="math_qa", category="math_mcq")
    assert rec is not None
    assert rec["mcq"]["choices"] == ["13", "42", "67", "76"]
    assert rec["messages"][1]["content"] == "B) 42"


def test_normalize_mcq_rejects_malformed():
    assert normalize_mcq({}, "s", "c") is None
    bad_key = dict(ARC_STYLE, answerKey="Z")
    assert normalize_mcq(bad_key, "s", "c") is None
    one_choice = dict(ARC_STYLE, choices={"text": ["Only"], "label": ["A"]})
    assert normalize_mcq(one_choice, "s", "c") is None


def test_normalize_qa_gsm8k_style():
    rec = normalize_qa({"question": "2+2?", "answer": "2+2 = 4. #### 4"}, "gsm8k", "math_qa")
    assert rec is not None and validate_record(rec) is None
    assert rec["messages"][1]["content"].endswith("#### 4")


def test_convert_records_streams_and_drops_bad():
    out = list(convert_records([ARC_STYLE, {}, MATHQA_STYLE], "mcq", "s", "c"))
    assert len(out) == 2
    assert {r["mcq"]["answer_idx"] for r in out} == {1}


def _tiny():
    return NueronceModel(NueronceConfig(
        byte_embed_dim=12, d_local=16, d_model=24, p_max=12, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=8, decoder_window=12,
        decoder_layers=1, d_state=6, channel_dim=8, min_patch=2, max_patch=10))


def test_rank_choices_returns_valid_index_and_eval_reports_chance():
    np.random.seed(0)
    model = _tiny()
    recs = [normalize_mcq(ARC_STYLE, "arc", "mcq"), normalize_mcq(MATHQA_STYLE, "mq", "mcq")]
    idx = rank_choices(model, recs[0]["messages"][0]["content"], recs[0]["mcq"]["choices"], max_len=160)
    assert 0 <= idx < 4
    res = evaluate_mcq(model, recs, max_len=160)
    assert res["n"] == 2 and abs(res["chance"] - 0.25) < 1e-9
    assert 0.0 <= res["accuracy"] <= 1.0
