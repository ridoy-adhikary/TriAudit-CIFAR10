#!/usr/bin/env python3
"""Phase 9: research report draft from logged metrics only."""

from __future__ import annotations

import json
from datetime import datetime

import common
from common import MONITOR, REPORTS, RUNS

from triaudit.utils import ensure_ridoy_or_warn


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL9 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ensure_ridoy_or_warn()
    table = json.loads((REPORTS / "tables" / "summary_mean_std.json").read_text(encoding="utf-8"))
    audit = json.loads((RUNS / "eval" / "audit_status.json").read_text(encoding="utf-8"))
    report = REPORTS / "TriAudit_research_report.md"

    lines = [
        "# TriAudit Research Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Methods",
        "",
        "- Dataset: CIFAR-10 with 45k/5k split; official test held out until final eval.",
        "- Model: CIFAR-adapted ResNet-18.",
        "- Conditions: Standard Training (ST) and PGD-AT (PGD-10, ε=8/255, α=2/255, random start).",
        "- Seeds: paired {0,1,2} with shared initialization.",
        "- Attacks: FGSM, PGD-20, multi-restart PGD (50 steps / 10 restarts) at primary ε=8/255; sensitivity at 4/255 and 12/255.",
        "- Calibration: ECE (15 equal-width bins), NLL, multiclass Brier on clean/FGSM/PGD-20.",
        "",
        "## Results (from runs/ artifacts)",
        "",
        "Mean ± std across seeds:",
        "",
    ]
    for k, v in table.items():
        lines.append(f"- **{k}**: {v['mean']:.4f} ± {v['std']:.4f} (values={v['values']})")

    lines += [
        "",
        "### Per-seed TriAudit Audit Status",
        "",
    ]
    for seed, payload in audit["seeds"].items():
        st = payload["triaudit_audit_status"]["status"]
        lines.append(f"- Seed {seed}: **{st}**")
        lines.append(f"  - measurements: `{payload['measurements']}`")

    lines += [
        "",
        "## Limitations and claim boundaries",
        "",
        "- CIFAR-10 only; single architecture; limited seeds.",
        "- White-box ℓ∞ threat model only.",
        "- Alert thresholds are project-defined, not universal standards.",
        "- Zero TriAudit flags does **not** mean the model is safe to deploy.",
        "- Missing metrics must never be treated as passing.",
        "",
        "## Artifact index",
        "",
        "- `runs/eval/attacks_summary.json`",
        "- `runs/eval/calibration_summary.json`",
        "- `runs/eval/audit_status.json`",
        "- `reports/tables/summary_mean_std.json`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
