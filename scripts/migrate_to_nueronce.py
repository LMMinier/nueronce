#!/usr/bin/env python3
"""One-time repository migration from legacy CFNA/MicroTorch naming to Nueronce.

Run from the repository root on a clean branch. The migration renames package
paths, public classes, scripts, tests, documentation, metrics labels, and data
prompts. It is intentionally deterministic and idempotent.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".txt",
    ".rst", ".ini", ".cfg", ".sh", ".ps1", ".bat", ".ipynb", ".csv",
    ".tsv", ".html", ".js", ".ts", ".cpp", ".cc", ".c", ".h", ".hpp",
}
PATH_REPLACEMENTS = (
    ("cfna_model", "nueronce_model"),
    ("cfna_blocks", "nueronce_blocks"),
    ("microtorch", "nueronce_engine"),
    ("MicroTorch", "NueronceEngine"),
    ("cfna", "nueronce"),
    ("CFNA", "NUERONCE"),
)
TEXT_REPLACEMENTS = (
    ("cfna.microtorch.cfna_model", "nueronce.engine.nueronce_model"),
    ("cfna.microtorch.cfna_blocks", "nueronce.engine.nueronce_blocks"),
    ("cfna.microtorch", "nueronce.engine"),
    ("cfna_model", "nueronce_model"),
    ("cfna_blocks", "nueronce_blocks"),
    ("MicroCFNAModel", "NueronceModel"),
    ("MicroCFNAConfig", "NueronceConfig"),
    ("CFNAConfig", "NueronceConfig"),
    ("MicroTorch", "Nueronce Engine"),
    ("microtorch", "engine"),
    ("CFNA", "NUERONCE"),
    ("cfna", "nueronce"),
)


def rename_tree() -> None:
    legacy_engine = ROOT / "cfna" / "microtorch"
    if legacy_engine.exists():
        legacy_engine.rename(ROOT / "cfna" / "engine")
    legacy_package = ROOT / "cfna"
    if legacy_package.exists():
        legacy_package.rename(ROOT / "nueronce")
    for path in sorted(ROOT.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        new_name = path.name
        for old, new in PATH_REPLACEMENTS:
            new_name = new_name.replace(old, new)
        if new_name != path.name and not path.with_name(new_name).exists():
            path.rename(path.with_name(new_name))


def rewrite_text() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        migrated = source
        for old, new in TEXT_REPLACEMENTS:
            migrated = migrated.replace(old, new)
        if migrated != source:
            path.write_text(migrated, encoding="utf-8")


def main() -> None:
    rename_tree()
    rewrite_text()
    print("Nueronce naming migration complete.")


if __name__ == "__main__":
    main()
