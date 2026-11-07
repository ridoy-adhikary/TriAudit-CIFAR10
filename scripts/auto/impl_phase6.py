#!/usr/bin/env python3
"""Phase 6: export calibration tables from attack-eval artifacts (ECE-15/NLL/Brier)."""

from __future__ import annotations

from datetime import datetime

import common
from common import MONITOR, RUNS

from triaudit.utils import ensure_ridoy_or_warn, save_json
import json


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL6 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ensure_ridoy_or_warn()
    summary = json.loads((RUNS / "eval" / "attacks_summary.json").read_text(encoding="utf-8"))
    cal = {}
    for key, res in summary["seeds"].items():
        primary = res["by_eps"][str(8 / 255)]
        cal[key] = {
            "clean": res["clean_calibration"],
            "fgsm": primary["fgsm"]["calibration"],
            "pgd20": primary["pgd20"]["calibration"],
        }
        assert cal[key]["clean"]["ece_bins"] == 15
    save_json(RUNS / "eval" / "calibration_summary.json", cal)
    log("calibration export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
