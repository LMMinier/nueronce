#!/usr/bin/env python3
"""Stage and commit only Section A changes."""
import subprocess
import sys

def run(cmd, shell=False):
    """Run command and return stdout."""
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

# Check status
print("=== Current Status ===")
status = run("git status --porcelain=v1", shell=True)
print(status)

# Show Section A changes
print("\n=== Section A Script Changes ===")
diff = run('git diff -- scripts/train_forgeloop_sft.py', shell=True)
# Count lines added with interruption logic
interrupt_lines = [line for line in diff.split('\n') if 'KeyboardInterrupt' in line or 'interrupted' in line]
print(f"Found {len(interrupt_lines)} interrupt-related lines")

print("\n=== Section A Test File ===")
test_diff = run('git diff -- tests/test_safe_interruption.py', shell=True)
print(f"Test file: {len(test_diff.split(chr(10)))} lines")

# Verify pre-existing changes are NOT included
print("\n=== Verifying nueronce/incremental.py not staged ===")
inc_status = run('git diff --name-status -- nueronce/incremental.py', shell=True)
if inc_status:
    print("ERROR: nueronce/incremental.py would be included!")
    sys.exit(1)
else:
    print("OK: nueronce/incremental.py not included")

print("\n=== READY TO STAGE ===")
print("Run: git add -- scripts/train_forgeloop_sft.py tests/test_safe_interruption.py")
print("Then: git commit -m 'Add safe interruption checkpointing for SFT'")
