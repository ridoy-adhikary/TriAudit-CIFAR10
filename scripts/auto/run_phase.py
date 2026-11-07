#!/usr/bin/env python3
"""
Run one TriAudit implementation phase with hard previous-phase verification.

Usage:
  conda/ridoy python scripts/auto/run_phase.py --phase N
  conda/ridoy python scripts/auto/run_phase.py --phase N --force-verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "project_monitor"
ACTIVITY = MONITOR / "project_activity.log"
RIDOY_PYTHON = Path("/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python")
AUTO = Path(__file__).resolve().parent

sys.path.insert(0, str(AUTO))
from phase_state import gate_passed, load_state, previous_verified, save_state  # noqa: E402


def log(msg: str) -> None:
    MONITOR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PHASE | {msg}"
    print(line, flush=True)
    with ACTIVITY.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def ensure_ridoy() -> None:
    exe = Path(sys.executable).resolve()
    expected = RIDOY_PYTHON.resolve()
    if exe != expected:
        # Re-exec under ridoy if possible
        if expected.is_file():
            log(f"re-exec under ridoy: {expected} (was {exe})")
            os.execv(str(expected), [str(expected), *sys.argv])
        raise SystemExit(f"Refusing to run outside ridoy env. Current={exe}")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    log("CMD " + " ".join(cmd))
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env)
    return proc.returncode


def verify_phase(phase: int) -> int:
    mapping = {
        0: AUTO / "verify_architecture.py",
        1: AUTO / "verify_phase1.py",
        2: AUTO / "verify_phase2.py",
        3: AUTO / "verify_phase3.py",
        4: AUTO / "verify_phase4.py",
        5: AUTO / "verify_phase5.py",
        6: AUTO / "verify_phase6.py",
        7: AUTO / "verify_phase7.py",
        8: AUTO / "verify_phase8.py",
        9: AUTO / "verify_phase9.py",
    }
    script = mapping[phase]
    if not script.is_file():
        log(f"FAIL missing verifier {script}")
        return 2
    return run_cmd([str(RIDOY_PYTHON), str(script)])


def implement_phase(phase: int) -> int:
    """Dispatch implementation entrypoints. Later phases call real train/eval scripts."""
    if phase == 0:
        return verify_phase(0)

    if phase == 1:
        # Scaffold + split + model smoke — implemented by prepare script when present
        prep = ROOT / "scripts" / "prepare_data.py"
        model_check = AUTO / "impl_phase1.py"
        if model_check.is_file():
            return run_cmd([str(RIDOY_PYTHON), str(model_check)])
        if prep.is_file():
            rc = run_cmd([str(RIDOY_PYTHON), str(prep)])
            if rc != 0:
                return rc
        # If impl not yet written, fail closed so orchestrator does not fake progress
        if not model_check.is_file():
            log("FAIL phase1 impl missing: scripts/auto/impl_phase1.py")
            return 2
        return 0

    impl = AUTO / f"impl_phase{phase}.py"
    if impl.is_file():
        return run_cmd([str(RIDOY_PYTHON), str(impl)])
    log(f"FAIL implementation missing for phase {phase}: {impl}")
    return 2


def require_cuda_for_training(phase: int) -> bool:
    """Return False if training should abort due to missing CUDA."""
    if phase not in {2, 3, 4}:
        return True
    code = (
        "import torch; import sys; "
        "sys.exit(0 if torch.cuda.is_available() else 1)"
    )
    rc = subprocess.run([str(RIDOY_PYTHON), "-c", code]).returncode
    if rc != 0 and os.environ.get("TRIAUDIT_ALLOW_CPU", "0") != "1":
        log(
            "FAIL CUDA unavailable. Training phases require GPU. "
            "Set TRIAUDIT_ALLOW_CPU=1 only for tiny smoke pilots."
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--force-verify-only", action="store_true")
    parser.add_argument("--skip-prev-check", action="store_true", help="DEBUG only; do not use in auto mode")
    args = parser.parse_args()
    phase = args.phase

    ensure_ridoy()
    MONITOR.mkdir(parents=True, exist_ok=True)

    if not args.skip_prev_check and not previous_verified(phase):
        log(f"BLOCKED phase {phase}: previous phase not verified")
        return 10

    state = load_state()
    state["current_phase"] = phase
    state["status"] = "running"
    state["phases"][str(phase)]["status"] = "running"
    state["phases"][str(phase)]["started_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    try:
        if not args.force_verify_only:
            if not require_cuda_for_training(phase):
                state = load_state()
                state["status"] = "blocked_no_cuda"
                state["phases"][str(phase)]["status"] = "blocked_no_cuda"
                state["phases"][str(phase)]["notes"] = "CUDA unavailable"
                save_state(state)
                return 3
            rc = implement_phase(phase)
            if rc != 0:
                state = load_state()
                state["status"] = "failed"
                state["phases"][str(phase)]["status"] = "failed"
                state["phases"][str(phase)]["notes"] = f"implement exit {rc}"
                save_state(state)
                log(f"FAIL phase {phase} implementation rc={rc}")
                return rc

        rc = verify_phase(phase)
        state = load_state()
        if rc == 0 and gate_passed(phase):
            state["phases"][str(phase)]["status"] = "verified"
            state["phases"][str(phase)]["verified"] = True
            state["phases"][str(phase)]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            state["status"] = "phase_verified"
            save_state(state)
            log(f"PASS phase {phase} verified")
            return 0

        state["phases"][str(phase)]["status"] = "verify_failed"
        state["phases"][str(phase)]["verified"] = False
        state["status"] = "verify_failed"
        save_state(state)
        log(f"FAIL phase {phase} verification rc={rc}")
        return rc or 1
    except Exception as exc:
        state = load_state()
        state["status"] = "error"
        state["phases"][str(phase)]["status"] = "error"
        state["phases"][str(phase)]["notes"] = str(exc)
        save_state(state)
        log(f"ERROR phase {phase}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
