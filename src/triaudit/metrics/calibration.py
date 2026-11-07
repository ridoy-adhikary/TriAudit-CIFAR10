"""ECE (equal-width), NLL, multiclass Brier."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nll_score(probs: torch.Tensor, y: torch.Tensor) -> float:
    logp = torch.log(probs.clamp_min(1e-12))
    return float(F.nll_loss(logp, y, reduction="mean").item())


def brier_score(probs: torch.Tensor, y: torch.Tensor, num_classes: int = 10) -> float:
    onehot = F.one_hot(y, num_classes=num_classes).float()
    return float(((probs - onehot) ** 2).sum(dim=1).mean().item())


def ece_equal_width(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    ece = torch.zeros((), device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        inside = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
        if i == 0:
            inside = (conf >= bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
        if inside.any():
            acc = correct[inside].mean()
            avg_conf = conf[inside].mean()
            ece = ece + inside.float().mean() * (acc - avg_conf).abs()
    return float(ece.item())


def calibration_bundle(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> dict:
    return {
        "ece": ece_equal_width(probs, y, n_bins=n_bins),
        "nll": nll_score(probs, y),
        "brier": brier_score(probs, y),
        "n": int(y.numel()),
        "ece_bins": n_bins,
    }
