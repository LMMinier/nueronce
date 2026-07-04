#!/usr/bin/env python3
"""enwik8: the external yardstick. Downloads the standard 100MB Wikipedia
byte corpus, makes the canonical 90/5/5 split, and either (a) emits a
ByteCorpus-compatible directory so the existing trainers can train on it, or
(b) evaluates a checkpoint's bits-per-byte on the held-out test split.

Until a CFNA checkpoint is TRAINED on enwik8's train split, any number from
--eval on other-corpus checkpoints is a zero-shot transfer bpb — report it
as that, never as an enwik8 benchmark result.

Usage (Colab):
    python scripts/eval_enwik8.py --make-corpus corpus_enwik8      # once
    python scripts/train_checkpoint.py --preset base_35m --corpus corpus_enwik8 \
        --minutes 170 --amp --resume --out checkpoints/cfna_enwik8_35m.pt
    python scripts/eval_enwik8.py --eval checkpoints/cfna_enwik8_35m.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

LN2 = math.log(2.0)
URL = "http://mattmahoney.net/dc/enwik8.zip"
SPLIT = (90_000_000, 5_000_000, 5_000_000)  # standard train/valid/test bytes


def fetch(cache: Path) -> bytes:
    raw = cache / "enwik8"
    if not raw.exists():
        cache.mkdir(parents=True, exist_ok=True)
        z = cache / "enwik8.zip"
        if not z.exists():
            print(f"downloading {URL} ...")
            urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as f:
            f.extract("enwik8", cache)
    data = raw.read_bytes()
    assert len(data) == 100_000_000, len(data)
    return data


def make_corpus(out_dir: Path, cache: Path) -> None:
    """Emit a ByteCorpus-compatible dir (manifest.jsonl + text files), with
    the canonical byte split mapped to train/val (test kept separate)."""
    data = fetch(cache)
    docs = out_dir / "text"
    docs.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    records = []
    spans = {"train": (0, SPLIT[0]), "val": (SPLIT[0], SPLIT[0] + SPLIT[1]),
             "test": (SPLIT[0] + SPLIT[1], sum(SPLIT))}
    for split, (a, b) in spans.items():
        path = docs / f"enwik8_{split}.txt"
        path.write_bytes(data[a:b])
        if split == "test":
            continue  # ByteCorpus sees train/val; test stays for --eval only
        records.append({
            "document_id": f"enwik8_{split}", "title": f"enwik8 ({split})",
            "author": "Wikipedia contributors", "document_type": "benchmark",
            "source_collection": "enwik8", "source_locator": URL,
            "files_page": URL, "license": "CC BY-SA (Wikipedia dump excerpt)",
            "license_id": "cc-by-sa", "commercial_use": True,
            "attribution_required": True, "language": "en",
            "publication_year": 2006, "retrieved_at": today,
            "content_hash": "sha256:" + hashlib.sha256(data[a:b]).hexdigest(),
            "quality_score": 1.0, "n_bytes": b - a, "n_docs": 1,
            "split": split, "bucket": "benchmark", "phase": 1,
            "role": "benchmark", "path": str(path.relative_to(out_dir)),
        })
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    print(f"corpus -> {out_dir} (train 90MB, val 5MB; test 5MB reserved)")


def evaluate(ckpt: str, cache: Path, seq: int, batch: int, limit_mb: float) -> None:
    import numpy as np
    import torch
    from cfna.chat import load_checkpoint

    model, payload = load_checkpoint(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    data = fetch(cache)
    test = data[SPLIT[0] + SPLIT[1]: sum(SPLIT)]
    if limit_mb:
        test = test[: int(limit_mb * 1e6)]
    arr = np.frombuffer(test, dtype=np.uint8).astype(np.int64)
    n = (len(arr) - 1) // seq * seq
    windows = arr[:n].reshape(-1, seq)
    losses = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            chunk = torch.from_numpy(windows[i:i + batch]).to(device)
            losses.append(float(model.lm_loss(chunk).item()) * chunk.shape[0])
    bpb = sum(losses) / len(windows) / LN2
    trained_on = payload.get("corpus", "unknown")
    print(json.dumps({"checkpoint": ckpt, "enwik8_test_bpb": round(bpb, 4),
                      "bytes_evaluated": int(n), "seq": seq,
                      "trained_on": str(trained_on),
                      "note": ("TRAINED-ON-ENWIK8 result" if "enwik8" in str(trained_on)
                               else "ZERO-SHOT transfer — not an enwik8 benchmark result")},
                     indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/enwik8_cache")
    ap.add_argument("--make-corpus", default="", metavar="OUT_DIR")
    ap.add_argument("--eval", default="", metavar="CHECKPOINT")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-mb", type=float, default=0.0,
                    help="evaluate only the first N MB of test (0 = all 5MB)")
    args = ap.parse_args()
    cache = Path(args.cache)
    if args.make_corpus:
        make_corpus(Path(args.make_corpus), cache)
    if args.eval:
        evaluate(args.eval, cache, args.seq, args.batch, args.limit_mb)
    if not args.make_corpus and not args.eval:
        raise SystemExit("pass --make-corpus DIR and/or --eval CHECKPOINT")


if __name__ == "__main__":
    main()
