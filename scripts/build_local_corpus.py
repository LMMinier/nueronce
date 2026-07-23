#!/usr/bin/env python3
"""Build a ByteCorpus-compatible training corpus entirely from the texts
already committed to this repository -- no network, no Hugging Face.

For environments where HF/Gutenberg are unreachable (proxied sandboxes,
air-gapped boxes): the repo carries ~24 MB of owner-curated public-domain
books at the root plus the psych/, math/, and grammar directories. This
script cleans them, splits by WHOLE DOCUMENT into train/val (so held-out
bits/byte measures generalization to unseen books, per corpus/dataset.py's
fairness rules), and writes corpus_dir/manifest.jsonl in the exact shape
nueronce.corpus.dataset.ByteCorpus loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SKIP_DIRS = {".git", "checkpoints", "corpus", "corpus_large", "data", "runs",
             "metrics", "artifacts", "docs", "tests", "scripts", "notebooks",
             "nueronce", "cfna", "schemas", "benchmarks", "dist", "build",
             "__pycache__", ".pytest_cache"}
SKIP_NAMES = {"requirements.txt", "checksums.txt", "train.jsonl", "val.jsonl",
              "test.jsonl", "manifest.jsonl"}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()


def deterministic_split(key: str, val_fraction: float) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < val_fraction else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_local")
    ap.add_argument("--root", default=".")
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--min-bytes", type=int, default=20_000)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out)
    (out / "text").mkdir(parents=True, exist_ok=True)

    records = []
    n_train = n_val = 0
    for path in sorted(root.rglob("*.txt")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.name.lower() in SKIP_NAMES:
            continue
        try:
            text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if len(text.encode("utf-8")) < args.min_bytes:
            continue
        split = deterministic_split(rel.as_posix(), args.val_fraction)
        doc_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", rel.as_posix())[:80] + ".txt"
        doc_path = out / "text" / doc_name
        doc_path.write_text(text, encoding="utf-8")
        records.append({
            "title": path.stem,
            "path": f"text/{doc_name}",
            "split": split,
            "source": f"repo:{rel.as_posix()}",
            "license": "Public domain (owner-curated classic texts)",
            "n_bytes": len(text.encode("utf-8")),
        })
        if split == "val":
            n_val += 1
        else:
            n_train += 1

    if n_val == 0 and records:   # guarantee at least one held-out document
        records.sort(key=lambda r: r["n_bytes"])
        records[0]["split"] = "val"
        n_val, n_train = 1, n_train - 1

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    total = sum(r["n_bytes"] for r in records)
    val_bytes = sum(r["n_bytes"] for r in records if r["split"] == "val")
    print(f"{len(records)} documents / {total/1e6:.1f} MB "
          f"({n_train} train, {n_val} val = {val_bytes/1e6:.1f} MB held-out) -> {out}")
    for r in records:
        if r["split"] == "val":
            print(f"  held out: {r['title']} ({r['n_bytes']/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
