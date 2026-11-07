#!/usr/bin/env python3
"""Verify Phase 3 acceptance artifacts (architecture Milestone 3)."""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "project_monitor"
OUT = MONITOR / "phase3_verify.json"
ACTIVITY = MONITOR / "project_activity.log"
REQUIRED = ['runs/pgd_at_pilot/metrics.json', 'tests/test_pgd_constraints.py']

def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] VERIFY3 | {msg}"
    print(line, flush=True)
    with ACTIVITY.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def main() -> int:
    checks = []
    ok = True
    for rel in REQUIRED:
        p = (ROOT / rel).exists()
        checks.append({"check": rel, "pass": p})
        ok = ok and p
    result = {
        "phase": 3,
        "name": "phase3",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "passed": ok,
        "checks": checks,
        "note": "Fails closed until architecture acceptance artifacts exist",
    }
    MONITOR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"{'PASS' if ok else 'FAIL'} {OUT}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
