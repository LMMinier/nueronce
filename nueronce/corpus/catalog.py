"""Load and validate the expanded NUERONCE corpus catalog.

The catalog contains metadata and rights-routing instructions, not downloaded book
text. Source adapters must still resolve item-level rights before acquisition.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = _REPO_ROOT / "corpus" / "catalog"
WORKS_DIR = CATALOG_DIR / "works"

LICENSE_BUCKETS = frozenset({
    "A_PD_CC0",
    "B_ATTRIBUTION",
    "C_SHARE_ALIKE",
    "D_NONCOMMERCIAL",
    "E_PER_ITEM",
    "F_REFERENCE_ONLY",
})

# Canonicalize two source identifiers that arrived with transcription artifacts.
SOURCE_ID_ALIASES = {
    "d oaj": "doaj",
    "bсcampus_opened": "bccampus_opened",  # source contains a Cyrillic small c
}


@dataclass(frozen=True)
class CollectionEntry:
    source_id: str
    name: str
    domain: str
    homepage: str
    license_bucket: str
    ingest_status: str
    recommended_adapter: str


@dataclass(frozen=True)
class WorkEntry:
    work_id: str
    title: str
    author: str
    publication_year: int | None
    subject: str
    work_type: str
    collection_code: str
    rights_code: str
    priority: str


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_unique(values: Iterable[str], label: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} values in corpus catalog")


def load_collections(path: Path | None = None) -> list[CollectionEntry]:
    path = path or (CATALOG_DIR / "collections.csv")
    entries: list[CollectionEntry] = []
    for row in _rows(path):
        source_id = SOURCE_ID_ALIASES.get(row["source_id"], row["source_id"])
        bucket = row["license_bucket"]
        if bucket not in LICENSE_BUCKETS:
            raise ValueError(f"unknown license bucket {bucket!r} for {source_id!r}")
        entries.append(CollectionEntry(
            source_id=source_id,
            name=row["name"],
            domain=row["domain"],
            homepage=row["homepage"],
            license_bucket=bucket,
            ingest_status=row["ingest_status"],
            recommended_adapter=row["recommended_adapter"],
        ))
    _require_unique((entry.source_id for entry in entries), "source_id")
    return entries


def load_works(directory: Path | None = None) -> list[WorkEntry]:
    directory = directory or WORKS_DIR
    paths = sorted(directory.glob("works_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no work catalog shards found under {directory}")

    entries: list[WorkEntry] = []
    for path in paths:
        for row in _rows(path):
            year = row["publication_year"].strip()
            entries.append(WorkEntry(
                work_id=row["work_id"],
                title=row["title"],
                author=row["author"],
                publication_year=int(year) if year else None,
                subject=row["subject"],
                work_type=row["work_type"],
                collection_code=row["collection_code"],
                rights_code=row["rights_code"],
                priority=row["priority"],
            ))
    entries.sort(key=lambda entry: entry.work_id)
    _require_unique((entry.work_id for entry in entries), "work_id")
    return entries


def validate_catalog() -> dict[str, object]:
    collections = load_collections()
    works = load_works()
    if len(collections) != 153:
        raise ValueError(f"expected 153 collections, found {len(collections)}")
    if len(works) != 289:
        raise ValueError(f"expected 289 works, found {len(works)}")

    bucket_counts: dict[str, int] = {}
    for entry in collections:
        bucket_counts[entry.license_bucket] = bucket_counts.get(entry.license_bucket, 0) + 1

    return {
        "collections": len(collections),
        "works": len(works),
        "license_buckets": dict(sorted(bucket_counts.items())),
        "work_shards": len(list(WORKS_DIR.glob("works_*.csv"))),
    }


if __name__ == "__main__":
    print(json.dumps(validate_catalog(), indent=2, sort_keys=True))


__all__ = [
    "CATALOG_DIR",
    "WORKS_DIR",
    "CollectionEntry",
    "WorkEntry",
    "load_collections",
    "load_works",
    "validate_catalog",
]
