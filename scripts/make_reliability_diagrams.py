#!/usr/bin/env python3
"""Reliability diagrams from frozen checkpoints (seed 0 representative)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-triaudit")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triaudit.attacks.pgd import fgsm, pgd
from triaudit.models.resnet_cifar import ResNet18CIFAR

FIG = ROOT / "reports" / "figures"
N_BINS = 15
EPS = 8 / 255
STEP = 2 / 255
COLORS = {"st": "#1f4e79", "at": "#c45c26"}


def load_model(ckpt_path: Path, device: torch.device) -> ResNet18CIFAR:
    model = ResNet18CIFAR(num_classes=10)
    blob = torch.load(ckpt_path, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def test_loader(batch_size: int = 256) -> DataLoader:
    ds = CIFAR10(
        root=str(ROOT / "dataset" / "raw"),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


@torch.no_grad()
def reliability_from_logits(logits: torch.Tensor, y: torch.Tensor, n_bins: int = N_BINS):
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_acc, bin_conf, bin_frac = [], [], []
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == 0:
            inside = (conf >= lo) & (conf <= hi)
        else:
            inside = (conf > lo) & (conf <= hi)
        if inside.any():
            bin_acc.append(float(correct[inside].mean().item()))
            bin_conf.append(float(conf[inside].mean().item()))
            bin_frac.append(float(inside.float().mean().item()))
        else:
            bin_acc.append(float("nan"))
            bin_conf.append(float((lo + hi) / 2))
            bin_frac.append(0.0)
    ece = 0.0
    for a, c, f in zip(bin_acc, bin_conf, bin_frac):
        if f > 0 and not np.isnan(a):
            ece += f * abs(a - c)
    return {
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_frac": bin_frac,
        "ece": ece,
        "acc": float(correct.mean().item()),
        "avg_conf": float(conf.mean().item()),
    }


def collect_condition(model, loader, device, condition: str):
    logits_all, y_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if condition == "clean":
            x_in = x
        elif condition == "fgsm":
            x_in = fgsm(model, x, y, eps=EPS)
        elif condition == "pgd20":
            x_in = pgd(model, x, y, eps=EPS, step_size=STEP, steps=20, random_start=True)
        else:
            raise ValueError(condition)
        with torch.no_grad():
            logits_all.append(model(x_in).detach())
            y_all.append(y.detach())
    return reliability_from_logits(torch.cat(logits_all), torch.cat(y_all))


def plot_reliability(ax, stats: dict, title: str, color: str):
    centers = np.linspace(1 / (2 * N_BINS), 1 - 1 / (2 * N_BINS), N_BINS)
    width = 1.0 / N_BINS
    frac = np.asarray(stats["bin_frac"])
    acc = np.asarray(stats["bin_acc"], dtype=float)
    # Gap bars: confidence bin mass shown as alpha via hatch overlay
    ax.bar(
        centers,
        np.nan_to_num(acc, nan=0.0),
        width=width * 0.95,
        color=color,
        alpha=0.75,
        edgecolor="white",
        label="bin accuracy",
    )
    # Ideal diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444", linewidth=1.1, label="perfect")
    # Average confidence vs accuracy markers
    ax.scatter([stats["avg_conf"]], [stats["acc"]], s=55, color="black", zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"{title}\nECE={stats['ece']:.3f}  acc={100*stats['acc']:.1f}%  conf={100*stats['avg_conf']:.1f}%",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--")
    # Show occupancy as secondary thin bars at bottom
    ax2 = ax.twinx()
    ax2.bar(centers, frac, width=width * 0.95, color="#999999", alpha=0.25)
    ax2.set_ylim(0, max(0.35, float(frac.max()) * 1.4 if frac.max() > 0 else 0.35))
    ax2.set_ylabel("bin mass", fontsize=8, color="#666")
    ax2.tick_params(axis="y", labelsize=7, colors="#666")
    ax2.spines["top"].set_visible(False)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    loader = test_loader(batch_size=256)
    models = {
        "ST": load_model(ROOT / "runs" / "seed0" / "st" / "checkpoint.pt", device),
        "PGD-AT": load_model(ROOT / "runs" / "seed0" / "pgd_at" / "checkpoint.pt", device),
    }
    conditions = ("clean", "fgsm", "pgd20")
    results = {}
    for name, model in models.items():
        results[name] = {}
        for cond in conditions:
            print(f"computing {name} / {cond} ...", flush=True)
            results[name][cond] = collect_condition(model, loader, device, cond)
            print(f"  ECE={results[name][cond]['ece']:.4f}", flush=True)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.2))
    color_map = {"ST": COLORS["st"], "PGD-AT": COLORS["at"]}
    for r, name in enumerate(("ST", "PGD-AT")):
        for c, cond in enumerate(conditions):
            plot_reliability(
                axes[r, c],
                results[name][cond],
                f"{name} · {cond.upper() if cond != 'pgd20' else 'PGD-20'}",
                color_map[name],
            )
    fig.suptitle(
        "Reliability diagrams (seed 0, official CIFAR-10 test, ε=8/255)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    out = FIG / "reliability_diagrams_seed0.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Persist numeric bins for reproducibility
    serializable = {
        name: {
            cond: {k: (v if not isinstance(v, float) else v) for k, v in stats.items()}
            for cond, stats in by_cond.items()
        }
        for name, by_cond in results.items()
    }
    (FIG / "reliability_diagrams_seed0.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
