#!/usr/bin/env python3
"""Commit Section A changes."""
import subprocess
import sys
import json

def run(cmd):
    """Run git command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=r"C:\nueronce-claude-new-session-tkjr3b")
    return result.returncode, result.stdout.strip(), result.stderr.strip()

print("=== Section A Commit Process ===\n")

# Show what will be staged
print("FILES TO STAGE:")
code, out, err = run("git diff --name-only")
for f in out.split('\n'):
    if f:
        print(f"  - {f}")

print("\n=== STAGING SECTION A FILES ===")
code, out, err = run('git add -- scripts/train_forgeloop_sft.py tests/test_safe_interruption.py')
if code != 0:
    print(f"Error staging: {err}")
    sys.exit(1)

print("Staged successfully")

print("\n=== STAGED CHANGES ===")
code, out, err = run("git diff --cached --name-status")
for line in out.split('\n'):
    if line:
        print(line)

print("\n=== VERIFYING UNSTAGED FILES ===")
code, out, err = run("git diff --name-status")
print("Unstaged files:")
for line in out.split('\n'):
    if line:
        print(f"  {line}")
        
# Check that nueronce/incremental.py is still unstaged (pre-existing change)
if "nueronce/incremental.py" in out:
    print("✓ nueronce/incremental.py remains unstaged (pre-existing)")
else:
    print("✓ No pre-existing files staged")

print("\n=== COMMITTING ===")
code, out, err = run('git commit -m "Add safe interruption checkpointing for SFT"')
if code == 0:
    print(out)
    
    # Get commit hash
    code, hash_out, _ = run("git rev-parse HEAD")
    print(f"\nCommit hash: {hash_out}")
else:
    print(f"Error committing: {err}")
    sys.exit(1)

print("\n=== POST-COMMIT STATUS ===")
code, out, err = run("git status --porcelain=v1")
print("Remaining uncommitted changes:")
for line in out.split('\n'):
    if line:
        print(f"  {line}")

print("\n✓ Section A committed successfully")
