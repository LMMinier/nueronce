#!/usr/bin/env python3
"""Vendor verified public-domain book chunks from LMMinier/ELDRAE-.

This copies actual text files into ``corpus/raw/public_domain`` and writes a
SHA-256 provenance manifest. Only explicitly whitelisted works are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE_REPO = "https://github.com/LMMinier/ELDRAE-.git"
SOURCE_REF = "c70df164ba3b53db3fa5d101591f08ea9b4bbedc"

# These are public-domain works or U.S. government works already chunked in ELDRAE-.
PREFIXES = (
    "the-prince",
    "the-art-of-war",
    "federalist-papers",
    "darwin-origin-of-species",
    "the-critique-of-pure-reason",
    "aristotle-nicomachean-ethics",
    "rousseau-social-contract",
    "plato-the-republic",
    "epictetus-enchiridion",
    "john-stuart-mill-on-liberty",
    "william-james-principles-of-psychology-vol1",
    "william-james-principles-of-psychology-vol2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("corpus/raw/public_domain"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nueronce-corpus-") as tmp:
        checkout = Path(tmp) / "eldrae"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_REPO, str(checkout)],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", SOURCE_REF, "--", "brain/sources"], check=True)

        source_dir = checkout / "brain" / "sources"
        selected = [
            path for path in source_dir.glob("*.md")
            if path.name.startswith(PREFIXES)
        ]
        if not selected:
            raise RuntimeError("whitelist matched no source files")

        manifest = []
        for src in sorted(selected):
            dest = args.out / src.name
            shutil.copy2(src, dest)
            manifest.append({
                "path": str(dest.as_posix()),
                "sha256": sha256(dest),
                "source_repository": SOURCE_REPO,
                "source_ref": SOURCE_REF,
                "source_path": f"brain/sources/{src.name}",
                "rights_bucket": "A_PD_CC0",
            })

    manifest_path = args.out / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "files": len(manifest),
        "bytes": sum((args.out / Path(row["path"]).name).stat().st_size for row in manifest),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
