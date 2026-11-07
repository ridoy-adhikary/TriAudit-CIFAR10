#!/usr/bin/env python3
"""Read/write TriAudit auto-implementation phase state."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "project_monitor"
STATE_PATH = MONITOR / "phase_state.json"

PHASES = [
    {"id": 0, "name": "architecture_verify", "gate": "phase0_architecture_verify.json"},
    {"id": 1, "name": "dataset_model_setup", "gate": "phase1_verify.json"},
    {"id": 2, "name": "st_pilot", "gate": "phase2_verify.json"},
    {"id": 3, "name": "pgd_at_pilot", "gate": "phase3_verify.json"},
    {"id": 4, "name": "paired_training", "gate": "phase4_verify.json"},
    {"id": 5, "name": "attack_evaluation", "gate": "phase5_verify.json"},
    {"id": 6, "name": "calibration_evaluation", "gate": "phase6_verify.json"},
    {"id": 7, "name": "audit_aggregation", "gate": "phase7_verify.json"},
    {"id": 8, "name": "stats_visualization", "gate": "phase8_verify.json"},
    {"id": 9, "name": "research_report", "gate": "phase9_verify.json"},
]


def default_state() -> dict[str, Any]:
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "current_phase": 0,
        "status": "idle",
        "env": "ridoy",
        "python": "/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python",
        "data_root": "dataset",
        "phases": {
            str(p["id"]): {
                "name": p["name"],
                "status": "pending",
                "verified": False,
                "gate_file": p["gate"],
                "started_at": None,
                "finished_at": None,
                "notes": "",
            }
            for p in PHASES
        },
    }


def load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return default_state()


def save_state(state: dict[str, Any]) -> None:
    MONITOR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def gate_passed(phase_id: int) -> bool:
    gate = MONITOR / PHASES[phase_id]["gate"]
    if not gate.is_file():
        return False
    try:
        data = json.loads(gate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(data.get("passed"))


def previous_verified(phase_id: int) -> bool:
    if phase_id <= 0:
        return True
    return gate_passed(phase_id - 1)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    state = load_state()
    if cmd == "init":
        state = default_state()
        save_state(state)
        print(STATE_PATH)
        return 0
    if cmd == "show":
        print(json.dumps(state, indent=2))
        return 0
    if cmd == "can-start":
        phase = int(sys.argv[2])
        allowed = previous_verified(phase)
        print(json.dumps({"phase": phase, "can_start": allowed, "previous_gate_ok": allowed}))
        return 0 if allowed else 1
    print("usage: phase_state.py [init|show|can-start N]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
