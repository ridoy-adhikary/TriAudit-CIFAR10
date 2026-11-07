#!/usr/bin/env python3
"""Generate TriAudit presentation figures from logged eval artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Writable matplotlib cache on shared servers
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-triaudit")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

EPS_KEYS = {
    4: "0.01568627450980392",
    8: "0.03137254901960784",
    12: "0.047058823529411764",
}
SEEDS = (0, 1, 2)
CONDITIONS = ("clean", "fgsm", "pgd20")

# Presentation style: clear, print-friendly, not purple-default
COLORS = {
    "st": "#1f4e79",
    "at": "#c45c26",
    "fgsm": "#2a6f6f",
    "pgd20": "#8b2942",
    "mr": "#5c6b73",
    "flag": "#b33a3a",
    "ok": "#2f6b4f",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def style_axes(ax, title: str, ylabel: str | None = None, xlabel: str | None = None):
    ax.set_title(title, fontsize=12, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def collect_attack_series(attacks: dict) -> dict:
    """Aggregate robust accuracy mean±std across seeds."""
    out = {
        "st_fgsm": {e: [] for e in EPS_KEYS},
        "st_pgd20": {e: [] for e in EPS_KEYS},
        "at_fgsm": {e: [] for e in EPS_KEYS},
        "at_pgd20": {e: [] for e in EPS_KEYS},
        "st_clean": [],
        "at_clean": [],
        "at_pgd20_primary": [],
        "at_mr_primary": [],
        "st_pgd20_primary": [],
        "st_mr_primary": [],
        "at_attack_gap": [],
        "st_attack_gap": [],
    }
    for seed in SEEDS:
        st = attacks["seeds"][f"seed{seed}_st"]
        at = attacks["seeds"][f"seed{seed}_pgd_at"]
        out["st_clean"].append(st["clean_acc"])
        out["at_clean"].append(at["clean_acc"])
        for e_int, e_key in EPS_KEYS.items():
            out["st_fgsm"][e_int].append(st["by_eps"][e_key]["fgsm"]["robust_acc"])
            out["st_pgd20"][e_int].append(st["by_eps"][e_key]["pgd20"]["robust_acc"])
            out["at_fgsm"][e_int].append(at["by_eps"][e_key]["fgsm"]["robust_acc"])
            out["at_pgd20"][e_int].append(at["by_eps"][e_key]["pgd20"]["robust_acc"])
        e8 = EPS_KEYS[8]
        out["st_pgd20_primary"].append(st["by_eps"][e8]["pgd20"]["robust_acc"])
        out["at_pgd20_primary"].append(at["by_eps"][e8]["pgd20"]["robust_acc"])
        out["st_mr_primary"].append(st["by_eps"][e8]["mr_pgd"]["robust_acc"])
        out["at_mr_primary"].append(at["by_eps"][e8]["mr_pgd"]["robust_acc"])
        out["st_attack_gap"].append(float(st["by_eps"][e8]["attack_gap"]))
        out["at_attack_gap"].append(float(at["by_eps"][e8]["attack_gap"]))
    return out


def fig_summary_bars(series: dict) -> Path:
    labels = ["Clean ST", "Clean AT", "PGD-20 AT"]
    means = [
        mean_std(series["st_clean"])[0],
        mean_std(series["at_clean"])[0],
        mean_std(series["at_pgd20_primary"])[0],
    ]
    stds = [
        mean_std(series["st_clean"])[1],
        mean_std(series["at_clean"])[1],
        mean_std(series["at_pgd20_primary"])[1],
    ]
    colors = [COLORS["st"], COLORS["at"], COLORS["pgd20"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.62, edgecolor="white")
    for b, m, s in zip(bars, means, stds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + s + 1.2,
            f"{m:.1f}±{s:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    style_axes(ax, "TriAudit summary (mean ± std over seeds 0–2)", ylabel="Accuracy (%)")
    return save(fig, "summary_bars.png")


def fig_clean_tradeoff(series: dict, audit: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: paired clean accuracy
    ax = axes[0]
    x = np.arange(len(SEEDS))
    w = 0.36
    st = [series["st_clean"][i] for i in range(3)]
    at = [series["at_clean"][i] for i in range(3)]
    ax.bar(x - w / 2, st, width=w, color=COLORS["st"], label="ST")
    ax.bar(x + w / 2, at, width=w, color=COLORS["at"], label="PGD-AT")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylim(70, 100)
    ax.legend(frameon=False)
    style_axes(ax, "Clean accuracy by seed", ylabel="Clean accuracy (%)")

    # Right: delta clean with threshold
    ax = axes[1]
    deltas = [audit["seeds"][str(s)]["layer1"]["delta_clean"] for s in SEEDS]
    thr = audit["seeds"]["0"]["layer1"]["threshold"]
    colors = [COLORS["flag"] if d < thr else COLORS["ok"] for d in deltas]
    ax.bar(x, deltas, color=colors, width=0.55)
    ax.axhline(thr, color="#444", linestyle="--", linewidth=1.2, label=f"flag threshold ({thr:g} pp)")
    ax.axhline(0, color="#888", linewidth=0.8)
    for i, d in enumerate(deltas):
        ax.text(i, d - 0.6, f"{d:.1f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax, "Layer 1: clean accuracy cost (AT − ST)", ylabel="Δ clean (pp)")

    fig.suptitle("Clean-data utility", fontsize=13, y=1.02)
    fig.tight_layout()
    return save(fig, "clean_accuracy_tradeoff.png")


def fig_robust_vs_eps(series: dict) -> Path:
    eps = np.array([4, 8, 12], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    def line(key_prefix: str, attack: str, label: str, color: str, ls: str, marker: str):
        means, stds = [], []
        for e in (4, 8, 12):
            m, s = mean_std(series[f"{key_prefix}_{attack}"][e])
            means.append(m)
            stds.append(s)
        means = np.asarray(means)
        stds = np.asarray(stds)
        ax.plot(eps, means, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=7, label=label)
        ax.fill_between(eps, means - stds, means + stds, color=color, alpha=0.12)

    line("st", "fgsm", "ST · FGSM", COLORS["st"], "--", "o")
    line("st", "pgd20", "ST · PGD-20", COLORS["st"], "-", "s")
    line("at", "fgsm", "PGD-AT · FGSM", COLORS["at"], "--", "o")
    line("at", "pgd20", "PGD-AT · PGD-20", COLORS["at"], "-", "s")

    ax.set_xticks([4, 8, 12])
    ax.set_xticklabels(["4/255", "8/255\n(primary)", "12/255"])
    ax.set_ylim(-2, 80)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style_axes(
        ax,
        "Robust accuracy vs ε (ℓ∞, mean ± std over seeds)",
        ylabel="Robust accuracy (%)",
        xlabel="Perturbation budget ε",
    )
    return save(fig, "robust_acc_vs_eps.png")


def fig_attack_adequacy(series: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    ax = axes[0]
    labels = ["ST", "PGD-AT"]
    x = np.arange(len(labels))
    w = 0.34
    pgd_m = [mean_std(series["st_pgd20_primary"])[0], mean_std(series["at_pgd20_primary"])[0]]
    pgd_s = [mean_std(series["st_pgd20_primary"])[1], mean_std(series["at_pgd20_primary"])[1]]
    mr_m = [mean_std(series["st_mr_primary"])[0], mean_std(series["at_mr_primary"])[0]]
    mr_s = [mean_std(series["st_mr_primary"])[1], mean_std(series["at_mr_primary"])[1]]
    ax.bar(x - w / 2, pgd_m, yerr=pgd_s, width=w, capsize=4, color=COLORS["pgd20"], label="PGD-20")
    ax.bar(x + w / 2, mr_m, yerr=mr_s, width=w, capsize=4, color=COLORS["mr"], label="Multi-restart PGD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 60)
    ax.legend(frameon=False)
    style_axes(ax, "Primary ε=8/255 attack comparison", ylabel="Robust accuracy (%)")

    ax = axes[1]
    gaps_st = series["st_attack_gap"]
    gaps_at = series["at_attack_gap"]
    xs = np.arange(len(SEEDS))
    ax.bar(xs - 0.18, gaps_st, width=0.34, color=COLORS["st"], label="ST AttackGap")
    ax.bar(xs + 0.18, gaps_at, width=0.34, color=COLORS["at"], label="AT AttackGap")
    thr = 5.0
    ax.axhline(thr, color="#444", linestyle="--", linewidth=1.2, label=f"flag if > {thr:g} pp")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax, "Layer 2: AttackGap = PGD-20 − MR-PGD", ylabel="AttackGap (pp)")

    fig.suptitle("Attack adequacy / white-box robustness", fontsize=13, y=1.02)
    fig.tight_layout()
    return save(fig, "attack_adequacy.png")


def fig_calibration_panels(calib: dict) -> Path:
    metrics = ("ece", "nll", "brier")
    titles = {"ece": "ECE (15 bins)", "nll": "NLL", "brier": "Brier score"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    x = np.arange(len(CONDITIONS))
    w = 0.36

    for ax, metric in zip(axes, metrics):
        st_means, st_stds, at_means, at_stds = [], [], [], []
        for cond in CONDITIONS:
            st_vals = [calib[f"seed{s}_st"][cond][metric] for s in SEEDS]
            at_vals = [calib[f"seed{s}_pgd_at"][cond][metric] for s in SEEDS]
            m, sd = mean_std(st_vals)
            st_means.append(m)
            st_stds.append(sd)
            m, sd = mean_std(at_vals)
            at_means.append(m)
            at_stds.append(sd)
        ax.bar(x - w / 2, st_means, yerr=st_stds, width=w, capsize=3, color=COLORS["st"], label="ST")
        ax.bar(x + w / 2, at_means, yerr=at_stds, width=w, capsize=3, color=COLORS["at"], label="PGD-AT")
        if metric == "ece":
            ax.axhline(0.15, color="#444", linestyle="--", linewidth=1.1, label="ECE flag 0.15")
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean", "FGSM", "PGD-20"])
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax, titles[metric], ylabel=metric.upper())

    fig.suptitle("Layer 3: confidence calibration (ε=8/255 attacks)", fontsize=13, y=1.03)
    fig.tight_layout()
    return save(fig, "calibration_panels.png")


def fig_clean_vs_robust(series: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for i, seed in enumerate(SEEDS):
        ax.scatter(
            series["st_clean"][i],
            series["st_pgd20_primary"][i],
            s=90,
            color=COLORS["st"],
            marker="o",
            zorder=3,
            label="ST" if i == 0 else None,
        )
        ax.scatter(
            series["at_clean"][i],
            series["at_pgd20_primary"][i],
            s=90,
            color=COLORS["at"],
            marker="D",
            zorder=3,
            label="PGD-AT" if i == 0 else None,
        )
        ax.annotate(
            f"s{seed}",
            (series["st_clean"][i], series["st_pgd20_primary"][i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=COLORS["st"],
        )
        ax.annotate(
            f"s{seed}",
            (series["at_clean"][i], series["at_pgd20_primary"][i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=COLORS["at"],
        )
        ax.plot(
            [series["st_clean"][i], series["at_clean"][i]],
            [series["st_pgd20_primary"][i], series["at_pgd20_primary"][i]],
            color="#bbbbbb",
            linewidth=1.0,
            zorder=1,
        )
    ax.set_xlim(78, 98)
    ax.set_ylim(-2, 55)
    ax.legend(frameon=False)
    style_axes(
        ax,
        "Utility–robustness trade-off (ε=8/255 PGD-20)",
        ylabel="PGD-20 robust accuracy (%)",
        xlabel="Clean accuracy (%)",
    )
    return save(fig, "clean_vs_robust_tradeoff.png")


def fig_audit_status(audit: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Left: layer flags heatmap-like bars
    ax = axes[0]
    layers = ["Layer1\nclean cost", "Layer2\nattack gap", "Layer3\ncalibration"]
    mat = np.zeros((3, 3))  # seeds x layers, 1=flag
    for i, seed in enumerate(SEEDS):
        s = audit["seeds"][str(seed)]
        mat[i, 0] = 1.0 if s["layer1"]["flag"] else 0.0
        mat[i, 1] = 1.0 if s["layer2"]["flag"] else 0.0
        mat[i, 2] = 1.0 if s["layer3"]["flag"] else 0.0
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"seed {s}" for s in SEEDS])
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                "FLAG" if mat[i, j] > 0.5 else "ok",
                ha="center",
                va="center",
                color="white" if mat[i, j] > 0.5 else "#1b4332",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_title("Per-layer flags", fontsize=12, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1], label="flag")

    # Right: overall status
    ax = axes[1]
    statuses = [audit["seeds"][str(s)]["triaudit_audit_status"]["status"] for s in SEEDS]
    n_flags = [audit["seeds"][str(s)]["triaudit_audit_status"]["n_flags"] for s in SEEDS]
    colors = [COLORS["flag"] if n >= 2 else ("#d4a017" if n == 1 else COLORS["ok"]) for n in n_flags]
    bars = ax.bar(range(3), n_flags, color=colors, width=0.55)
    for i, (b, st) in enumerate(zip(bars, statuses)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, st, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylim(0, 3.4)
    ax.set_yticks([0, 1, 2, 3])
    style_axes(ax, "TriAudit Audit Status (# flags)", ylabel="Number of layer flags")
    ax.text(
        0.02,
        0.98,
        "0 flags ≠ deploy-safe",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555",
    )

    fig.suptitle("TriAudit three-layer audit aggregation", fontsize=13, y=1.02)
    fig.tight_layout()
    return save(fig, "audit_status_overview.png")


def fig_methods_overview() -> Path:
    """Static methods schematic for slides (no numeric invention)."""
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, 1.8, 1.2, "CIFAR-10\n45k/5k + held-out test"),
        (2.4, 1.0, 1.8, 1.2, "ResNet-18\nCIFAR stem"),
        (4.5, 1.0, 1.8, 1.2, "Paired seeds\nST vs PGD-AT"),
        (6.6, 1.0, 1.5, 1.2, "Freeze\ncheckpoints"),
        (8.4, 1.0, 1.4, 1.2, "3-layer\naudit"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#f3f1ec", edgecolor="#333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for x0, x1 in [(2.1, 2.4), (4.2, 4.5), (6.3, 6.6), (8.1, 8.4)]:
        ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("TriAudit pipeline (methods overview)", fontsize=12, pad=6)
    ax.text(
        5.0,
        0.35,
        "Threat model: white-box untargeted ℓ∞, primary ε=8/255  |  Eval: FGSM, PGD-20, multi-restart PGD + ECE/NLL/Brier",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333",
    )
    return save(fig, "methods_pipeline.png")


def main() -> int:
    attacks = load_json(RUNS / "eval" / "attacks_summary.json")
    calib = load_json(RUNS / "eval" / "calibration_summary.json")
    audit = load_json(RUNS / "eval" / "audit_status.json")
    series = collect_attack_series(attacks)

    outputs = [
        fig_methods_overview(),
        fig_summary_bars(series),
        fig_clean_tradeoff(series, audit),
        fig_robust_vs_eps(series),
        fig_attack_adequacy(series),
        fig_calibration_panels(calib),
        fig_clean_vs_robust(series),
        fig_audit_status(audit),
    ]

    manifest = {
        "figures": [str(p.relative_to(ROOT)) for p in outputs],
        "source_artifacts": [
            "runs/eval/attacks_summary.json",
            "runs/eval/calibration_summary.json",
            "runs/eval/audit_status.json",
        ],
        "note": "All numeric plots are derived from logged runs/; no hand-typed metrics.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "figures" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Generated figures:")
    for p in outputs:
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
