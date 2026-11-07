#!/usr/bin/env python3
"""Verify Phase 1 acceptance criteria from architecture Milestone 1."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "project_monitor"
OUT = MONITOR / "phase1_verify.json"
ACTIVITY = MONITOR / "project_activity.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] VERIFY1 | {msg}"
    print(line, flush=True)
    with ACTIVITY.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    checks = []
    ok = True

    split = ROOT / "dataset" / "splits" / "cifar10_45k5k_seed42.json"
    split_ok = split.is_file()
    checks.append({"check": "split_file", "pass": split_ok})
    ok &= split_ok
    if split_ok:
        data = json.loads(split.read_text(encoding="utf-8"))
        tr, va = data["train_indices"], data["val_indices"]
        disjoint = set(tr).isdisjoint(va)
        sizes = len(tr) == 45000 and len(va) == 5000
        checks.append({"check": "split_sizes", "pass": sizes})
        checks.append({"check": "split_disjoint", "pass": disjoint})
        ok &= sizes and disjoint

    model_file = ROOT / "src" / "triaudit" / "models" / "resnet_cifar.py"
    checks.append({"check": "model_module", "pass": model_file.is_file()})
    ok &= model_file.is_file()

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from triaudit.models.resnet_cifar import ResNet18CIFAR

        m = ResNet18CIFAR()
        y = m(torch.randn(4, 3, 32, 32))
        shape_ok = tuple(y.shape) == (4, 10)
        checks.append({"check": "forward_shape", "pass": shape_ok, "detail": str(tuple(y.shape))})
        ok &= shape_ok
    except Exception as exc:
        checks.append({"check": "forward_shape", "pass": False, "detail": str(exc)})
        ok = False

    for rel in [
        "configs/data/cifar10.yaml",
        "configs/model/resnet18_cifar.yaml",
        "configs/train/st.yaml",
        "configs/train/pgd_at.yaml",
        "src/triaudit/data/cifar10.py",
    ]:
        p = (ROOT / rel).is_file()
        checks.append({"check": rel, "pass": p})
        ok &= p

    result = {
        "phase": 1,
        "name": "dataset_model_setup",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "passed": ok,
        "checks": checks,
    }
    MONITOR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"{'PASS' if ok else 'FAIL'} {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
