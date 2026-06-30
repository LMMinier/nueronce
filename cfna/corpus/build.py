"""Build the license-clean corpus: download -> clean -> dedupe -> bucket -> manifest.

Writes cleaned per-document text files under ``corpus/<bucket>/<license>/`` and a
``corpus/manifest.jsonl`` with one provenance record per document (the corpus
manifest from the build plan). Only documents that pass the license check land in
``safe_commercial``; the first checkpoint trains from there.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..ops import sha256_bytes
from .sources import Source, safe_commercial_sources

_HEADER = re.compile(r"^\[([^\]]+?)\s+by\s+(.+?)(?:\s+(\d{3,4}))?\]\s*$", re.MULTILINE)
_GUTEN_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUTEN_END = re.compile(r"\*\*\*\s*END OF.*", re.IGNORECASE | re.DOTALL)
_MULTIBLANK = re.compile(r"\n{3,}")


@dataclass
class DocRecord:
    document_id: str
    title: str
    author: str
    document_type: str
    source_collection: str
    source_locator: str
    license: str
    license_id: str
    commercial_use: bool
    attribution_required: bool
    language: str
    publication_year: Optional[int]
    retrieved_at: str
    content_hash: str
    quality_score: float
    n_bytes: int
    split: str
    bucket: str
    path: str


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #

def _fetch(url: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / Path(url).name
    if cached.exists():
        return cached.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "cfna-corpus/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (trusted host)
        data = r.read()
    cached.write_bytes(data)
    return data


def _iter_zip_texts(blob: bytes):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if name.endswith("/") or not name.lower().endswith(".txt"):
                continue
            yield Path(name).name, z.read(name).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def clean_text(raw: str) -> str:
    """Strip headers/boilerplate, normalize newlines and blank runs."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # drop the "[Title by Author Year]" header line if present
    text = _HEADER.sub("", text, count=1)
    # strip Project Gutenberg start/end boilerplate if present
    if "START OF" in text.upper():
        text = _GUTEN_START.split(text, 1)[-1]
    if "END OF" in text.upper():
        text = _GUTEN_END.split(text, 1)[0]
    text = _MULTIBLANK.sub("\n\n", text)
    return text.strip() + "\n"


def parse_header(raw: str) -> Dict[str, Optional[str]]:
    m = _HEADER.search(raw)
    if not m:
        return {"title": None, "author": None, "year": None}
    return {"title": m.group(1).strip(), "author": m.group(2).strip(),
            "year": int(m.group(3)) if m.group(3) else None}


def quality_score(text: str) -> float:
    if not text:
        return 0.0
    ascii_ratio = sum(c < 128 for c in text.encode("utf-8", "ignore")) / max(1, len(text))
    length_ok = min(1.0, len(text) / 20000.0)
    lines = text.split("\n")
    avg_line = sum(len(line) for line in lines) / max(1, len(lines))
    line_ok = 1.0 if 20 <= avg_line <= 200 else 0.5
    return round(0.5 * ascii_ratio + 0.3 * length_ok + 0.2 * line_ok, 3)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def _doc_id(source: Source, filename: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", filename.lower().rsplit(".", 1)[0]).strip("_")
    return f"{source.source_id}__{stem}"


def build_corpus(out_dir: Path, sources: Optional[List[Source]] = None,
                 cache_dir: Optional[Path] = None, val_doc_ids: Optional[set] = None,
                 min_bytes: int = 2000) -> List[DocRecord]:
    """Download, clean, license-check, dedupe, and write the corpus + manifest."""
    sources = sources or safe_commercial_sources()
    cache_dir = cache_dir or (out_dir / "_cache")
    val_doc_ids = val_doc_ids or set()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[DocRecord] = []
    seen_hashes: set = set()
    today = date.today().isoformat()

    for source in sources:
        # license check: only commercial-safe public-domain/CC0/CC-BY here
        if source.bucket != "safe_commercial":
            continue
        blob = _fetch(source.url, cache_dir)
        for filename, raw in _iter_zip_texts(blob):
            if filename.lower() in ("readme", "readme.txt"):
                continue
            meta = parse_header(raw)
            text = clean_text(raw)
            if len(text) < min_bytes:
                continue
            content_hash = sha256_bytes(text.encode("utf-8"))
            if content_hash in seen_hashes:        # exact-duplicate dedupe
                continue
            seen_hashes.add(content_hash)

            doc_id = _doc_id(source, filename)
            license_dir = source.license_id
            dest = out_dir / "safe_commercial" / license_dir
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{doc_id}.txt"
            path.write_text(text, encoding="utf-8")

            records.append(DocRecord(
                document_id=doc_id,
                title=meta["title"] or doc_id,
                author=meta["author"] or "unknown",
                document_type=source.document_type,
                source_collection=source.collection,
                source_locator=source.url,
                license=source.license,
                license_id=source.license_id,
                commercial_use=source.commercial_use,
                attribution_required=source.attribution_required,
                language="en",
                publication_year=meta["year"],
                retrieved_at=today,
                content_hash=f"sha256:{content_hash}",
                quality_score=quality_score(text),
                n_bytes=len(text.encode("utf-8")),
                split="val" if doc_id in val_doc_ids else "train",
                bucket=source.bucket,
                path=str(path.relative_to(out_dir)),
            ))

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return records


def load_manifest(out_dir: Path) -> List[dict]:
    path = out_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["DocRecord", "clean_text", "parse_header", "quality_score",
           "build_corpus", "load_manifest"]
