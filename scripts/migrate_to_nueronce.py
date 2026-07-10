#!/usr/bin/env python3
"""One-time repository migration from legacy project naming to Nueronce.

Run from the repository root on a clean branch. The migration renames package
paths, public classes, scripts, tests, documentation, metrics labels, and data
prompts. It is intentionally deterministic and idempotent.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PACKAGE = "c" + "fna"
LEGACY_PACKAGE_UPPER = "C" + "FNA"
LEGACY_ENGINE = "micro" + "torch"
LEGACY_ENGINE_TITLE = "Micro" + "Torch"
TEXT_EXTENSIONS = {
    ".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".txt",
    ".rst", ".ini", ".cfg", ".sh", ".ps1", ".bat", ".ipynb", ".csv",
    ".tsv", ".html", ".js", ".ts", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".tex", ".xml", ".log", ".lock", ".cff",
}
PATH_REPLACEMENTS = (
    (f"{LEGACY_PACKAGE}_model", "nueronce_model"),
    (f"{LEGACY_PACKAGE}_blocks", "nueronce_blocks"),
    (LEGACY_ENGINE, "nueronce_engine"),
    (LEGACY_ENGINE_TITLE, "NueronceEngine"),
    (LEGACY_PACKAGE, "nueronce"),
    (LEGACY_PACKAGE_UPPER, "NUERONCE"),
)
TEXT_REPLACEMENTS = (
    (f"{LEGACY_PACKAGE}.{LEGACY_ENGINE}.{LEGACY_PACKAGE}_model", "nueronce.engine.nueronce_model"),
    (f"{LEGACY_PACKAGE}.{LEGACY_ENGINE}.{LEGACY_PACKAGE}_blocks", "nueronce.engine.nueronce_blocks"),
    (f"{LEGACY_PACKAGE}.{LEGACY_ENGINE}", "nueronce.engine"),
    ("nueronce_model", "nueronce_model"),
    ("nueronce_blocks", "nueronce_blocks"),
    (f"Micro{LEGACY_PACKAGE_UPPER}Model", "NueronceModel"),
    (f"Micro{LEGACY_PACKAGE_UPPER}Config", "NueronceConfig"),
    (f"{LEGACY_PACKAGE_UPPER}Config", "NueronceConfig"),
    ("Nueronce Engine", "Nueronce Engine"),
    (LEGACY_ENGINE_TITLE, "Nueronce Engine"),
    (LEGACY_ENGINE, "nueronce_engine"),
    (LEGACY_PACKAGE_UPPER, "NUERONCE"),
    (LEGACY_PACKAGE, "nueronce"),
)


def rename_tree() -> None:
    legacy_engine = ROOT / LEGACY_PACKAGE / LEGACY_ENGINE
    if legacy_engine.exists():
        legacy_engine.rename(ROOT / LEGACY_PACKAGE / "engine")
    legacy_package = ROOT / LEGACY_PACKAGE
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
