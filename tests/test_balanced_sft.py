"""Balanced SFT curriculum tests."""

import json
import subprocess
import sys

from nueronce.training.balanced_sft import balanced_records, category_counts


def test_balanced_records_equalize_category_budget():
    rows = list(balanced_records(examples_per_category=3, categories=["greetings", "facts"]))
    counts = category_counts(rows)
    assert counts == {"greetings": 3, "facts": 3}
    assert all(r["id"].startswith("balanced-") for r in rows)


def test_balanced_builder_writes_leakage_free_splits(tmp_path):
    out = tmp_path / "balanced"
    cmd = [
        sys.executable, "scripts/build_balanced_sft_dataset.py",
        "--out-dir", str(out),
        "--train-examples-per-category", "20",
        "--val-per-category", "2",
        "--test-per-category", "2",
        "--num-shards", "2",
        "--examples-per-shard", "100",
    ]
    subprocess.run(cmd, check=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["split"]["n_train"] == 200
    assert manifest["split"]["n_validation"] > 0
    assert manifest["balanced_dataset"]["train_repetition_is_weighting_not_new_data"] is True
