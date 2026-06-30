#!/usr/bin/env python3
"""Build the license-clean public-domain corpus and print a manifest summary.

Usage:  python scripts/build_corpus.py [--out corpus]

Holds out a few whole documents for validation (generalization to unseen books).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from cfna.corpus.build import build_corpus

# Whole documents held out for validation (never trained on).
VAL_DOCS = {
    "gutenberg_classics__carroll_alice",
    "gutenberg_classics__shakespeare_macbeth",
    "us_inaugural_addresses__1861_lincoln",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="corpus")
    args = ap.parse_args()

    out = Path(args.out)
    records = build_corpus(out, val_doc_ids=VAL_DOCS)

    total = sum(r.n_bytes for r in records)
    by_type = Counter(r.document_type for r in records)
    by_split = Counter(r.split for r in records)
    print(f"built {len(records)} documents, {total/1e6:.2f} MB clean text -> {out}/")
    print(f"  types: {dict(by_type)}   splits: {dict(by_split)}")
    print(f"  licenses: {dict(Counter(r.license_id for r in records))}")
    print("\n  document                                  type     bytes   split  license")
    for r in sorted(records, key=lambda r: -r.n_bytes):
        print(f"  {r.document_id:<40} {r.document_type:<7} {r.n_bytes:>8}  {r.split:<5}  {r.license_id}")
    print(f"\nmanifest: {out}/manifest.jsonl")
    print("all documents are public domain (commercial-safe); bucket = safe_commercial")


if __name__ == "__main__":
    main()
