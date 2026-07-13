from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path


def load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_step_reads_engine_metadata(tmp_path):
    mod = load_script("run_nueronce_355m_to_convergence.py")
    path = tmp_path / "checkpoint.pkl"
    with path.open("wb") as f:
        pickle.dump({"meta": {"global_step": 270}}, f)
    assert mod.checkpoint_step(path) == 270


def test_jsonl_records_filters_invalid_and_other_events(tmp_path):
    mod = load_script("run_nueronce_355m_to_convergence.py")
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"event": "train", "step": 1}),
            "not-json",
            json.dumps({"event": "validation", "step": 2, "heldout_bpb": 4.2}),
        ]),
        encoding="utf-8",
    )
    assert mod.jsonl_records(path, "validation") == [
        {"event": "validation", "step": 2, "heldout_bpb": 4.2}
    ]


def test_sft_shards_are_sorted_and_required(tmp_path):
    mod = load_script("train_nueronce_engine_355m_sft_converge.py")
    (tmp_path / "shard_02.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "shard_01.jsonl").write_text("{}\n", encoding="utf-8")
    assert [p.name for p in mod.list_shards(tmp_path)] == ["shard_01.jsonl", "shard_02.jsonl"]


def test_job_pid_check_rejects_nonpositive_values():
    mod = load_script("nueronce_355m_job.py")
    assert mod._pid_running(0) is False
    assert mod._pid_running(-1) is False
