"""Large-scale dialogue dataset pipeline: JSONL parsing, validation, dedup,
deterministic splitting, sharding, and leakage checks. No PyTorch needed."""

import json

import numpy as np
import pytest

from cfna.training.dataset_prep import (
    assert_no_leakage, build_clean_dataset, record_normalized_hash,
    record_pair_hash, record_text_hash, split_and_shard, validate_record,
    write_manifest,
)


def _msg(role, content):
    return {"role": role, "content": content}


def _rec(rid, prompt, response, category="test"):
    return {"id": rid, "messages": [_msg("user", prompt), _msg("assistant", response)],
            "source": "unit-test", "category": category}


# --------------------------------------------------------------------------- #
# 1. JSONL parsing / round-trip
# --------------------------------------------------------------------------- #

def test_records_round_trip_through_jsonl(tmp_path):
    recs = [_rec("a", "Hi", "Hello!"), _rec("b", "Bye", "Goodbye!")]
    path = tmp_path / "clean.jsonl"
    stats = build_clean_dataset(recs, str(path))
    assert stats.accepted == 2
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert {p["id"] for p in parsed} == {"a", "b"}


# --------------------------------------------------------------------------- #
# 2. Invalid-record rejection
# --------------------------------------------------------------------------- #

def test_invalid_records_are_rejected():
    assert validate_record({"id": "x"}) is not None                       # missing fields
    assert validate_record({"id": "", "messages": [], "source": "s", "category": "c"}) is not None
    assert validate_record(_rec("x", "", "reply")) is not None             # empty content... wait content empty
    bad_role = {"id": "x", "messages": [{"role": "bot", "content": "hi"}], "source": "s", "category": "c"}
    assert validate_record(bad_role) is not None
    no_assistant = {"id": "x", "messages": [_msg("user", "hi")], "source": "s", "category": "c"}
    assert validate_record(no_assistant) is not None
    ends_on_user = {"id": "x", "messages": [_msg("user", "hi"), _msg("assistant", "hey"), _msg("user", "again")],
                    "source": "s", "category": "c"}
    assert validate_record(ends_on_user) is not None
    assert validate_record(_rec("ok", "Hi", "Hello")) is None


def test_build_clean_dataset_drops_invalid_and_counts_reasons(tmp_path):
    recs = [_rec("good", "Hi", "Hello!"), {"id": "bad"}, {"not": "a valid schema at all but still a dict"}]
    stats = build_clean_dataset(recs, str(tmp_path / "clean.jsonl"))
    assert stats.accepted == 1
    assert stats.rejected_invalid == 2
    assert sum(stats.rejection_reasons.values()) == 2


# --------------------------------------------------------------------------- #
# 3. Exact deduplication
# --------------------------------------------------------------------------- #

def test_exact_duplicates_are_removed(tmp_path):
    a = _rec("a", "Hi", "Hello!")
    b = _rec("b", "Hi", "Hello!")  # different id, identical content
    c = _rec("c", "Bye", "Goodbye!")
    stats = build_clean_dataset([a, b, c], str(tmp_path / "clean.jsonl"))
    assert stats.accepted == 2
    assert stats.rejected_exact_dup == 1


def test_near_duplicates_and_pair_duplicates_are_removed(tmp_path):
    a = _rec("a", "Hi there!", "Hello!!")
    near = _rec("b", "hi there", "hello")  # same after normalization
    stats = build_clean_dataset([a, near], str(tmp_path / "clean.jsonl"))
    assert stats.accepted == 1
    assert stats.rejected_near_dup == 1

    assert record_text_hash(a) != record_text_hash(near)
    assert record_normalized_hash(a) == record_normalized_hash(near)
    assert record_pair_hash(a) == record_pair_hash(near)


# --------------------------------------------------------------------------- #
# 4 + helpers: no split leakage
# --------------------------------------------------------------------------- #

def _build_and_split(tmp_path, n=200, val_size=20, test_size=20, num_shards=2, per_shard=50, seed=0):
    recs = [_rec(f"r{i}", f"prompt number {i}", f"response number {i}") for i in range(n)]
    clean = tmp_path / "clean.jsonl"
    build_clean_dataset(recs, str(clean))
    split = split_and_shard(str(clean), str(tmp_path), num_shards=num_shards,
                            examples_per_shard=per_shard, val_size=val_size,
                            test_size=test_size, seed=seed)
    return split


def test_no_leakage_between_train_val_test(tmp_path):
    split = _build_and_split(tmp_path)
    assert_no_leakage(split.train_shard_paths, split.val_path, split.test_path)  # raises on failure


def test_leakage_check_actually_detects_overlap(tmp_path):
    rec = _rec("dup", "same prompt", "same response")
    val_path = tmp_path / "val.jsonl"
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    val_path.write_text(json.dumps(rec) + "\n")
    test_path.write_text(json.dumps(_rec("t", "other prompt", "other response")) + "\n")
    train_path.write_text(json.dumps(rec) + "\n" + json.dumps(_rec("other", "x", "y")) + "\n")
    with pytest.raises(AssertionError):
        assert_no_leakage([str(train_path)], str(val_path), str(test_path))


# --------------------------------------------------------------------------- #
# 5. Ten-shard creation (parametrized on shard count, not hardcoded to 10)
# --------------------------------------------------------------------------- #

def test_shard_creation_produces_exact_requested_shards_and_sizes(tmp_path):
    split = _build_and_split(tmp_path, n=300, val_size=20, test_size=20, num_shards=5, per_shard=40)
    assert len(split.train_shard_paths) == 5
    assert split.n_train == 200
    for p in split.train_shard_paths:
        lines = [l for l in open(p) if l.strip()]
        assert len(lines) == 40


def test_split_raises_if_not_enough_clean_records(tmp_path):
    recs = [_rec(f"r{i}", f"p{i}", f"a{i}") for i in range(10)]
    clean = tmp_path / "clean.jsonl"
    build_clean_dataset(recs, str(clean))
    with pytest.raises(ValueError):
        split_and_shard(str(clean), str(tmp_path), num_shards=2, examples_per_shard=10,
                        val_size=5, test_size=5, seed=0)


# --------------------------------------------------------------------------- #
# 6. Deterministic shuffling
# --------------------------------------------------------------------------- #

def test_split_is_deterministic_given_the_same_seed(tmp_path):
    split1 = _build_and_split(tmp_path / "a", seed=123)
    split2 = _build_and_split(tmp_path / "b", seed=123)
    ids1 = [json.loads(l)["id"] for l in open(split1.val_path)]
    ids2 = [json.loads(l)["id"] for l in open(split2.val_path)]
    assert ids1 == ids2


def test_different_seeds_give_different_shuffles(tmp_path):
    split1 = _build_and_split(tmp_path / "a", seed=1)
    split2 = _build_and_split(tmp_path / "b", seed=2)
    ids1 = [json.loads(l)["id"] for l in open(split1.val_path)]
    ids2 = [json.loads(l)["id"] for l in open(split2.val_path)]
    assert ids1 != ids2


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def test_manifest_records_counts_and_category_distribution(tmp_path):
    recs = [_rec(f"r{i}", f"p{i}", f"a{i}", category="alpha" if i % 2 else "beta") for i in range(200)]
    clean = tmp_path / "clean.jsonl"
    stats = build_clean_dataset(recs, str(clean))
    split = split_and_shard(str(clean), str(tmp_path), num_shards=2, examples_per_shard=40,
                            val_size=20, test_size=20, seed=0)
    manifest = write_manifest(str(tmp_path), stats, split,
                              source_description="unit test", license_description="n/a")
    assert manifest["split"]["n_train"] == 80
    assert manifest["split"]["num_shards"] == 2
    assert set(manifest["split"]["train_category_counts"]) <= {"alpha", "beta"}
    assert (tmp_path / "manifest.json").exists()
