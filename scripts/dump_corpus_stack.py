#!/usr/bin/env python3
"""Dump staged NUERONCE training corpora into the ByteCorpus manifest format.

The script starts with the user's requested stack: TinyStories first, then
Cosmopedia-100k, then bounded streamed educational corpora. It records source
URLs, licenses, phase/role, hashes, and byte counts in ``manifest.jsonl`` so the
training script can consume the output directly.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional

from nueronce.corpus.stack import CORPUS_STACK, CorpusStackEntry, entries_for_phase, get_entry

_END = "\n\n<|END_DOCUMENT|>\n\n"
_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def row_text(entry: CorpusStackEntry, row: dict) -> Optional[str]:
    if entry.document_template == "dolly":
        instruction = str(row.get("instruction") or "").strip()
        context = str(row.get("context") or "").strip()
        response = str(row.get("response") or "").strip()
        if not instruction or not response:
            return None
        parts = [f"Instruction: {instruction}"]
        if context:
            parts.append(f"Context: {context}")
        parts.append(f"Response: {response}")
        return "\n".join(parts)

    if entry.document_template == "oasst1":
        if row.get("lang") and row.get("lang") != "en":
            return None
        text = str(row.get("text") or "").strip()
        role = row.get("role") or row.get("message_id") or "message"
        return f"{role}: {text}" if text else None

    for field in entry.text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def iter_huggingface(entry: CorpusStackEntry):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The 'datasets' package is required for Hugging Face corpus dumping. "
            "Install with: python -m pip install datasets"
        ) from exc

    kwargs = {"path": entry.dataset_name, "split": entry.split, "streaming": entry.streaming}
    if entry.dataset_config:
        kwargs["name"] = entry.dataset_config
    return load_dataset(**kwargs)


def selected_entries(args) -> list[CorpusStackEntry]:
    if args.sources:
        return [get_entry(source_id.strip()) for source_id in args.sources.split(",") if source_id.strip()]
    return [entry for entry in entries_for_phase(args.phase) if entry.loader == "huggingface"]


def write_records(entries: list[CorpusStackEntry], out_dir: Path, target_bytes: int,
                  max_docs_per_source: Optional[int], min_chars: int,
                  val_every: int = 20) -> list[dict]:
    docs_dir = out_dir / "stack_text"
    docs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    total_bytes = 0
    today = date.today().isoformat()

    for entry in entries:
        if entry.loader != "huggingface":
            print(f"skip {entry.source_id}: loader={entry.loader} needs a dedicated extractor")
            continue
        print(f"dump {entry.source_id}: {entry.dataset_name or entry.dataset_page}")
        split_files = {
            "train": docs_dir / f"{entry.source_id}__train.txt",
            "val": docs_dir / f"{entry.source_id}__val.txt",
        }
        split_docs = {"train": 0, "val": 0}
        split_bytes = {"train": 0, "val": 0}
        source_docs = 0
        handles = {split: path.open("w", encoding="utf-8") for split, path in split_files.items()}
        try:
            for idx, row in enumerate(iter_huggingface(entry)):
                text = row_text(entry, row)
                if not text:
                    continue
                text = clean_text(text)
                if len(text) < min_chars:
                    continue
                encoded = text.encode("utf-8")
                if total_bytes + len(encoded) > target_bytes:
                    break
                if max_docs_per_source is not None and source_docs >= max_docs_per_source:
                    break
                split = "val" if val_every > 0 and (source_docs + 1) % val_every == 0 else "train"
                handles[split].write(text)
                handles[split].write(_END)
                source_docs += 1
                split_docs[split] += 1
                split_bytes[split] += len(encoded)
                total_bytes += len(encoded)
        finally:
            for fh in handles.values():
                fh.close()
        if source_docs == 0:
            for path in split_files.values():
                path.unlink(missing_ok=True)
            continue
        for split, source_file in split_files.items():
            if split_docs[split] == 0:
                source_file.unlink(missing_ok=True)
                continue
            digest = sha256(source_file.read_bytes()).hexdigest()
            records.append({
                "document_id": f"{entry.source_id}_{split}",
                "title": f"{entry.name} ({split})",
                "author": "dataset",
                "document_type": entry.role,
                "source_collection": entry.name,
                "source_locator": entry.dataset_page,
                "files_page": entry.files_page,
                "license": entry.license,
                "license_id": entry.license,
                "commercial_use": "noncommercial" not in entry.license.lower(),
                "attribution_required": any(token in entry.license.lower() for token in ("by", "sharing", "share")),
                "language": "en",
                "publication_year": None,
                "retrieved_at": today,
                "content_hash": f"sha256:{digest}",
                "quality_score": 1.0,
                "n_bytes": split_bytes[split],
                "n_docs": split_docs[split],
                "split": split,
                "bucket": f"phase_{entry.phase}_{entry.role}",
                "phase": entry.phase,
                "role": entry.role,
                "path": str(source_file.relative_to(out_dir)),
            })
        print(
            f"  wrote {source_docs:,} docs / "
            f"{(split_bytes['train'] + split_bytes['val'])/1e6:.2f} MB "
            f"(train docs={split_docs['train']:,}, val docs={split_docs['val']:,})"
        )
        if total_bytes >= target_bytes:
            break
    return records


def write_stack_catalog(out_dir: Path) -> None:
    catalog = []
    for entry in CORPUS_STACK:
        item = asdict(entry)
        item["text_fields"] = list(entry.text_fields)
        catalog.append(item)
    (out_dir / "corpus_stack_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_stack", help="Output corpus directory")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Dump Hugging Face sources through this phase")
    ap.add_argument("--sources", default="",
                    help="Comma-separated source IDs, e.g. tinystories,cosmopedia_100k")
    ap.add_argument("--target-bytes", type=int, default=250_000_000,
                    help="Stop after this many UTF-8 bytes across selected sources")
    ap.add_argument("--max-docs-per-source", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=20,
                    help="Route every Nth accepted document to validation")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_stack_catalog(out_dir)
    entries = selected_entries(args)
    records = write_records(entries, out_dir, args.target_bytes, args.max_docs_per_source, args.min_chars, args.val_every)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"manifest: {manifest}")
    print(f"total: {len(records)} source files / {sum(r['n_bytes'] for r in records)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
