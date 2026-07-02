#!/usr/bin/env python3
"""Convert NUERONCE raw public-domain corpus to training JSONL."""
from __future__ import annotations

import argparse, json
from pathlib import Path


def convert(root: Path, out: Path) -> dict:
    manifest=root/"manifest.jsonl"
    records=[json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    total=0
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            text=Path(rec["path"]).read_text(encoding="utf-8")
            item={"id":rec["id"],"title":rec["title"],"author":rec.get("author"),"subject":rec.get("subject","unspecified"),"text":text,"source":rec.get("source_name"),"license":"public-domain-us"}
            fh.write(json.dumps(item, ensure_ascii=False)+"\n"); total+=len(text.encode("utf-8"))
    return {"documents":len(records),"bytes":total,"output":str(out)}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root", default="corpus/raw/public_domain"); ap.add_argument("--out", default="data/processed/public_domain/documents.jsonl"); args=ap.parse_args()
    print(json.dumps(convert(Path(args.root), Path(args.out)), indent=2))

if __name__ == "__main__": main()
