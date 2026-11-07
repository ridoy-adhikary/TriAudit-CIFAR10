#!/usr/bin/env python3
"""Phase 7: TriAudit audit aggregation per seed + summary."""

from __future__ import annotations

import json
from datetime import datetime

import common
from common import MONITOR, RUNS

from triaudit.metrics.audit_layers import (
    audit_status,
    layer1_clean_gap,
    layer2_attack_gap,
    layer3_calibration,
)
from triaudit.utils import ensure_ridoy_or_warn, save_json


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL7 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ensure_ridoy_or_warn()
    attacks = json.loads((RUNS / "eval" / "attacks_summary.json").read_text(encoding="utf-8"))
    cal = json.loads((RUNS / "eval" / "calibration_summary.json").read_text(encoding="utf-8"))
    eps = str(8 / 255)
    out = {"seeds": {}, "note": "Zero flags != safe to deploy"}

    for seed in (0, 1, 2):
        st = attacks["seeds"][f"seed{seed}_st"]
        at = attacks["seeds"][f"seed{seed}_pgd_at"]
        l1 = layer1_clean_gap(at["clean_acc"], st["clean_acc"])
        # Attack adequacy primarily on AT model (robustness claims)
        at_eps = at["by_eps"][eps]
        if "mr_pgd" not in at_eps:
            l2 = {"attack_gap": None, "flag": None, "unevaluated": True}
            l2_flag = None
        else:
            l2 = layer2_attack_gap(at_eps["pgd20"]["robust_acc"], at_eps["mr_pgd"]["robust_acc"])
            l2_flag = l2["flag"]
        eces = {
            "clean": cal[f"seed{seed}_pgd_at"]["clean"]["ece"],
            "fgsm": cal[f"seed{seed}_pgd_at"]["fgsm"]["ece"],
            "pgd20": cal[f"seed{seed}_pgd_at"]["pgd20"]["ece"],
        }
        l3 = layer3_calibration(eces)
        status = audit_status([l1["flag"], l2_flag, l3["flag"]])
        out["seeds"][str(seed)] = {
            "layer1": l1,
            "layer2": l2,
            "layer3": l3,
            "triaudit_audit_status": status,
            "measurements": {
                "clean_st": st["clean_acc"],
                "clean_at": at["clean_acc"],
                "pgd20_at": at_eps["pgd20"]["robust_acc"],
                "mr_at": at_eps.get("mr_pgd", {}).get("robust_acc"),
                "ece": eces,
            },
        }
    save_json(RUNS / "eval" / "audit_status.json", out)
    log("audit aggregation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
