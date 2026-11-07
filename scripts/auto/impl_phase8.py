#!/usr/bin/env python3
"""Phase 8: mean±std tables and simple figures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import common
from common import MONITOR, REPORTS, RUNS

from triaudit.utils import ensure_ridoy_or_warn, save_json


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL8 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def main() -> int:
    ensure_ridoy_or_warn()
    audit = json.loads((RUNS / "eval" / "audit_status.json").read_text(encoding="utf-8"))
    attacks = json.loads((RUNS / "eval" / "attacks_summary.json").read_text(encoding="utf-8"))
    eps = str(8 / 255)

    clean_st, clean_at, rob_at, gap, ece_clean = [], [], [], [], []
    for seed in ("0", "1", "2"):
        m = audit["seeds"][seed]["measurements"]
        clean_st.append(m["clean_st"])
        clean_at.append(m["clean_at"])
        rob_at.append(m["pgd20_at"])
        if m["mr_at"] is not None:
            gap.append(m["pgd20_at"] - m["mr_at"])
        ece_clean.append(m["ece"]["clean"])

    table = {}
    for name, vals in [
        ("clean_st", clean_st),
        ("clean_at", clean_at),
        ("delta_clean", [a - s for a, s in zip(clean_at, clean_st)]),
        ("pgd20_at", rob_at),
        ("attack_gap_at", gap),
        ("ece_clean_at", ece_clean),
    ]:
        mu, sd = mean_std(vals)
        table[name] = {"mean": mu, "std": sd, "values": vals}

    out_dir = REPORTS / "tables"
    fig_dir = REPORTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "summary_mean_std.json", table)

    # Minimal matplotlib figure if available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["clean_st", "clean_at", "pgd20_at"]
        means = [table[k]["mean"] for k in labels]
        stds = [table[k]["std"] for k in labels]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, means, yerr=stds, capsize=4)
        plt.ylabel("Accuracy (%)")
        plt.title("TriAudit summary (mean ± std over seeds)")
        plt.tight_layout()
        plt.savefig(fig_dir / "summary_bars.png", dpi=150)
        plt.close()
    except Exception as exc:
        log(f"plot skipped: {exc}")

    log("stats/visualization complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
