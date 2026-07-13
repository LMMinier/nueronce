#!/usr/bin/env python3
"""Start or inspect the detached NUERONCE 355M convergence job.

Examples:

    python scripts/nueronce_355m_job.py start -- \
      --source-checkpoint checkpoints/.../source_step270.pkl \
      --corpus corpus --build-sft-if-missing

    python scripts/nueronce_355m_job.py status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPO_ROOT / "runs" / "nueronce_355m_convergence"


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        return str(pid) in proc.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start(args: argparse.Namespace, controller_args: list[str]) -> None:
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    pid_path = workspace / "job.json"
    state_path = workspace / "state.json"
    log_path = Path(args.log).resolve() if args.log else workspace / "training.log"

    if pid_path.exists():
        try:
            old = json.loads(pid_path.read_text(encoding="utf-8"))
            old_pid = int(old.get("pid", 0))
            if _pid_running(old_pid):
                raise SystemExit(f"job already running with PID {old_pid}")
        except json.JSONDecodeError:
            pass

    controller = REPO_ROOT / "scripts" / "run_nueronce_355m_to_convergence.py"
    cmd = [
        sys.executable,
        str(controller),
        "--workspace", str(workspace),
        *controller_args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab", buffering=0)
    kwargs = {
        "cwd": str(REPO_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    record = {
        "pid": proc.pid,
        "command": cmd,
        "workspace": str(workspace),
        "state": str(state_path),
        "log": str(log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    pid_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


def status(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).resolve()
    pid_path = workspace / "job.json"
    state_path = workspace / "state.json"
    record = json.loads(pid_path.read_text(encoding="utf-8")) if pid_path.exists() else {}
    pid = int(record.get("pid", 0))
    payload = {
        "running": _pid_running(pid),
        "pid": pid or None,
        "workspace": str(workspace),
        "log": record.get("log"),
        "state": json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None,
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    start_ap = sub.add_parser("start")
    start_ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    start_ap.add_argument("--log", default="")
    start_ap.add_argument("controller_args", nargs=argparse.REMAINDER)

    status_ap = sub.add_parser("status")
    status_ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))

    args = ap.parse_args()
    if args.command == "start":
        controller_args = list(args.controller_args)
        if controller_args and controller_args[0] == "--":
            controller_args = controller_args[1:]
        start(args, controller_args)
    else:
        status(args)


if __name__ == "__main__":
    main()
