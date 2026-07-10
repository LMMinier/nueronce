#!/usr/bin/env python3
"""Publish a NUERONCE checkpoint as a verified GitHub Release asset.

The script intentionally keeps model weights out of normal Git history. It
checks the checkpoint against the committed manifest before creating/updating a
release and uploading the binary checkpoint plus its manifest.

Requires the GitHub CLI (`gh`) to be installed and authenticated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def manifest_value(manifest: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in manifest:
            return manifest[name]
    result = manifest.get("result")
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
    meta = manifest.get("meta")
    if isinstance(meta, dict):
        for name in names:
            if name in meta:
                return meta[name]
    return None


def verify_checkpoint(checkpoint: Path, manifest_path: Path) -> tuple[dict[str, Any], str]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest_value(manifest, "checkpoint_sha256", "sha256")
    expected_size = manifest_value(manifest, "checkpoint_bytes", "size_bytes")
    actual_hash = sha256(checkpoint)
    actual_size = checkpoint.stat().st_size

    if expected_hash and actual_hash.lower() != str(expected_hash).lower():
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if expected_size is not None and actual_size != int(expected_size):
        raise ValueError(
            f"checkpoint size mismatch: expected {expected_size}, got {actual_size}"
        )
    return manifest, actual_hash


def release_notes(manifest: dict[str, Any], checkpoint: Path, digest: str) -> str:
    model = manifest_value(manifest, "model") or "NUERONCE checkpoint"
    step = manifest_value(manifest, "checkpoint_step", "step")
    objective = manifest_value(manifest, "training_objective", "objective") or "unspecified"
    phase = manifest_value(manifest, "phase") or "unspecified"
    params = manifest_value(manifest, "parameters")
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}

    lines = [
        f"# {model}",
        "",
        "Verified model checkpoint distributed as a GitHub Release asset rather than a Git blob.",
        "",
        f"- **Checkpoint:** `{checkpoint.name}`",
        f"- **Step:** `{step if step is not None else 'unknown'}`",
        f"- **Objective:** `{objective}`",
        f"- **Phase:** `{phase}`",
        f"- **Parameters:** `{params if params is not None else 'unknown'}`",
        f"- **Bytes:** `{checkpoint.stat().st_size}`",
        f"- **SHA-256:** `{digest}`",
    ]
    for key in ("loss", "grad_norm", "grad_tensors", "forward_s", "backward_s", "update_s"):
        if key in result:
            lines.append(f"- **{key.replace('_', ' ').title()}:** `{result[key]}`")
    lines.extend([
        "",
        "## Integrity",
        "",
        "The publisher verifies the local checkpoint against the manifest before upload. Consumers should verify the SHA-256 after download.",
        "",
        "## Maturity",
        "",
        "A released checkpoint is a reproducible training artifact, not evidence by itself of language competence, reasoning, or benchmark performance. Consult the accompanying manifest and repository evaluations.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", default="LMMinier/nueronce")
    parser.add_argument("--target", default="claude/new-session-tkjr3b")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--extra", action="append", type=Path, default=[])
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--prerelease", action="store_true")
    args = parser.parse_args()

    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI 'gh' is not installed or not on PATH")
    run(["gh", "auth", "status"])

    manifest, digest = verify_checkpoint(args.checkpoint, args.manifest)
    step = manifest_value(manifest, "checkpoint_step", "step")
    default_tag = f"base35m-step{int(step):06d}" if step is not None else f"checkpoint-{digest[:12]}"
    tag = args.tag or default_tag
    title = args.title or f"NUERONCE base-35M step {step if step is not None else digest[:12]}"

    for extra in args.extra:
        if not extra.is_file():
            raise FileNotFoundError(extra)

    existing = run(["gh", "release", "view", tag, "--repo", args.repo], check=False)
    notes = release_notes(manifest, args.checkpoint, digest)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as handle:
        handle.write(notes)
        notes_path = Path(handle.name)

    try:
        if existing.returncode != 0:
            command = [
                "gh", "release", "create", tag,
                "--repo", args.repo,
                "--target", args.target,
                "--title", title,
                "--notes-file", str(notes_path),
            ]
            if args.draft:
                command.append("--draft")
            if args.prerelease:
                command.append("--prerelease")
            run(command)
        else:
            run([
                "gh", "release", "edit", tag,
                "--repo", args.repo,
                "--title", title,
                "--notes-file", str(notes_path),
            ])

        assets = [args.checkpoint, args.manifest, *args.extra]
        run([
            "gh", "release", "upload", tag,
            *(str(path) for path in assets),
            "--repo", args.repo,
            "--clobber",
        ])
        view = run([
            "gh", "release", "view", tag,
            "--repo", args.repo,
            "--json", "url,tagName,isDraft,isPrerelease,assets",
        ])
        payload = json.loads(view.stdout)
        payload["verified_checkpoint_sha256"] = digest
        payload["verified_checkpoint_bytes"] = args.checkpoint.stat().st_size
        print(json.dumps(payload, indent=2))
    finally:
        notes_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"release publication failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
