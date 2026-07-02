#!/usr/bin/env python3
"""Validate NUERONCE public-domain raw corpus artifacts."""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

REQUIRED = {"id","slug","title","author","path","source_name","source_url","source_identifier","rights_bucket","rights_basis","sha256","bytes"}
ROOT = Path("corpus/raw/public_domain")


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(root: Path = ROOT) -> tuple[list[str], dict]:
    errors=[]
    manifest=root/"manifest.jsonl"
    if not manifest.exists():
        return [f"missing manifest: {manifest}"], {}
    records=load_manifest(manifest)
    hashes=Counter(); titles=Counter(); source_ids=Counter(); subjects=Counter(); sources=Counter(); total=0
    for i, rec in enumerate(records, 1):
        missing=REQUIRED-set(rec)
        if missing: errors.append(f"record {i} missing fields: {sorted(missing)}")
        rel=Path(rec.get("path", ""))
        path=rel if rel.is_absolute() else Path(rel)
        try:
            resolved=path.resolve()
            root_res=root.resolve()
            if root_res not in resolved.parents and resolved != root_res:
                errors.append(f"record {i} path escapes corpus root: {path}")
        except Exception as exc:
            errors.append(f"record {i} invalid path {path}: {exc}"); continue
        if not path.exists():
            errors.append(f"record {i} missing text file: {path}"); continue
        data=path.read_bytes()
        if not data.strip(): errors.append(f"record {i} empty text: {path}")
        digest=hashlib.sha256(data).hexdigest()
        if rec.get("sha256") != digest: errors.append(f"record {i} sha256 mismatch: {path}")
        if rec.get("bytes") != len(data): errors.append(f"record {i} byte mismatch: {path}")
        if not rec.get("rights_bucket") or not rec.get("rights_basis"):
            errors.append(f"record {i} missing rights metadata")
        hashes[digest]+=1; titles[rec.get("title","")]+=1; source_ids[rec.get("source_identifier","")]+=1
        subjects[rec.get("subject","unspecified")]+=1; sources[rec.get("source_name","unspecified")]+=1; total+=len(data)
    for label, counter in (("hash",hashes),("title",titles),("source_identifier",source_ids)):
        for value, count in counter.items():
            if value and count>1: errors.append(f"duplicate {label}: {value!r} ({count})")
    return errors, {"records":len(records),"bytes":total,"subjects":dict(subjects),"sources":dict(sources)}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--root", default=str(ROOT)); args=ap.parse_args()
    errors, summary=validate(Path(args.root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        for e in errors: print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())
