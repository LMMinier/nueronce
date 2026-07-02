import json
import subprocess
import sys


def test_prompt_aligned_builder_writes_leakage_free_splits(tmp_path):
    out = tmp_path / "prompt_aligned"
    subprocess.run([
        sys.executable, "scripts/build_prompt_aligned_sft.py",
        "--out-dir", str(out),
        "--direct", "40",
        "--grounded", "40",
        "--edge", "20",
        "--num-shards", "2",
        "--examples-per-shard", "50",
        "--seed", "44",
    ], check=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["unique_total"] == 100
    assert manifest["curriculum_proportions_unique"] == {
        "direct": 40,
        "grounded": 40,
        "abstain_conflict_revision": 20,
    }
    assert all(v == 0 for v in manifest["leakage"].values())


def test_inference_phase2_suite_counts(tmp_path):
    suite = tmp_path / "phase2.jsonl"
    subprocess.run([
        sys.executable, "scripts/eval_inference_phase2.py",
        "--write-suite",
        "--suite", str(suite),
    ], check=True)
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert len(rows) == 230
    assert counts["grounded"] == 30
    assert counts["insufficient_evidence"] == 20
    assert counts["coding_explanation"] == 10


def test_torch_batch_preserves_evidence_retrieval_tensors():
    import torch
    from scripts.train_sft import _torch_batch_from_records

    records = [{
        "system_message": "sys",
        "user_request": "question",
        "trusted_evidence": ["[doc1] decisive evidence"],
        "response_plan": ["use doc1"],
        "assistant_response": "answer",
    }]
    batch = _torch_batch_from_records(records, max_len=128, device=torch.device("cpu"))
    assert batch["byte_ids"].shape[0] == 1
    assert batch["target_mask"].any()
    assert batch["neighbor_ids"] is not None
    assert batch["neighbor_mask"].any()
