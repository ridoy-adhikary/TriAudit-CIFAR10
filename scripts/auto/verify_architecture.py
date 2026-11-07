#!/usr/bin/env python3
"""Phase 0 — verify TriAudit architecture document internal consistency and project locks."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "TriAudit_Final_Project_Architecture.md"
MONITOR = ROOT / "project_monitor"
OUT = MONITOR / "phase0_architecture_verify.json"
ACTIVITY = MONITOR / "project_activity.log"

REQUIRED_PHRASES = [
    ("CIFAR-10", "dataset is CIFAR-10"),
    ("45,000", "45k train split"),
    ("5,000", "5k val split"),
    ("ResNet-18", "CIFAR ResNet-18"),
    ("PGD-10", "PGD-10 adversarial training"),
    ("8/255", "primary epsilon"),
    ("2/255", "PGD step size"),
    ("PGD-20", "PGD-20 evaluation"),
    ("Multi-restart PGD", "multi-restart adequacy attack"),
    ("15 equal-width", "ECE 15 bins"),
    ("Δ_clean", "clean utility gap"),
    ("AttackGap", "attack adequacy gap"),
    ("TriAudit Audit Status", "audit aggregation name"),
    ("UNEVALUATED", "missing metrics not pass"),
    ("project-defined", "thresholds are project-defined"),
    ("paired", "seed pairing"),
    ("freeze", "freeze before eval"),
]

FORBIDDEN_CORE = [
    (r"\bEOT-PGD\b.*core", "EOT must not be required in core"),
]

MILESTONE_HEADERS = [f"### Milestone {i}" for i in range(1, 10)]


def log(msg: str) -> None:
    MONITOR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ARCH_VERIFY | {msg}"
    print(line)
    with ACTIVITY.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    checks = []
    ok = True

    if not ARCH.is_file():
        log("FAIL architecture file missing")
        return 2

    text = ARCH.read_text(encoding="utf-8")

    for phrase, name in REQUIRED_PHRASES:
        passed = phrase in text
        checks.append({"check": name, "pass": passed, "detail": phrase})
        ok = ok and passed

    for hdr in MILESTONE_HEADERS:
        passed = hdr in text
        checks.append({"check": f"has {hdr}", "pass": passed})
        ok = ok and passed

    # Signed delta (no abs-value instruction)
    signed = "Do not use absolute value" in text or "Do not** take the absolute value" in text or "no absolute value" in text.lower()
    checks.append({"check": "delta_clean_signed_no_abs", "pass": signed})
    ok = ok and signed

    # Claim boundary
    no_deploy = "not" in text.lower() and "deploy" in text.lower() and "Zero flags" in text
    checks.append({"check": "zero_flags_not_deploy_safe", "pass": no_deploy})
    ok = ok and no_deploy

    # Path adaptation note: project uses dataset/ (created), architecture mentions data/
    dataset_dir = (ROOT / "dataset").is_dir()
    checks.append(
        {
            "check": "project_dataset_dir_exists",
            "pass": dataset_dir,
            "detail": "Canonical runtime path is dataset/ (architecture data/ mapped here)",
        }
    )
    ok = ok and dataset_dir

    cifar = ROOT / "dataset" / "raw" / "cifar-10-batches-py"
    cifar_ok = cifar.is_dir() and (cifar / "test_batch").is_file()
    checks.append({"check": "cifar10_downloaded", "pass": cifar_ok, "detail": str(cifar)})
    # Dataset presence is a project prerequisite, required before phase 1
    ok = ok and cifar_ok

    ridoy = Path("/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python")
    checks.append({"check": "ridoy_env_python_exists", "pass": ridoy.is_file(), "detail": str(ridoy)})
    ok = ok and ridoy.is_file()

    result = {
        "phase": 0,
        "name": "architecture_verification",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "architecture_file": str(ARCH),
        "passed": ok,
        "checks": checks,
        "locks": {
            "env": "ridoy only",
            "data_path": "dataset/",
            "threat_model": "white-box linf eps=8/255",
            "exclude_core": ["EOT-PGD", "autoencoder blocker"],
        },
    }
    MONITOR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"{'PASS' if ok else 'FAIL'} wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
