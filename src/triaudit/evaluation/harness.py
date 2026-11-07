"""Frozen-model attack + calibration evaluation harness."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from triaudit.attacks.pgd import fgsm, multi_restart_pgd, pgd, validate_constraints
from triaudit.metrics.calibration import calibration_bundle


@torch.no_grad()
def _collect_clean(model, loader, device):
    probs_all, y_all, correct = [], [], 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
        probs_all.append(probs.cpu())
        y_all.append(y.cpu())
    return (
        100.0 * correct / max(total, 1),
        torch.cat(probs_all),
        torch.cat(y_all),
    )


def evaluate_attack_loader(
    model: nn.Module,
    loader,
    device,
    attack_name: str,
    eps: float,
    step_size: float = 2 / 255,
    steps: int = 20,
    restarts: int = 1,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    viol = 0
    probs_all, y_all = [], []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if attack_name == "clean":
            x_adv = x
        elif attack_name == "fgsm":
            x_adv = fgsm(model, x, y, eps=eps)
        elif attack_name == "pgd20":
            x_adv = pgd(
                model, x, y, eps=eps, step_size=step_size, steps=steps, random_start=True
            )
        elif attack_name == "mr_pgd":
            x_adv = multi_restart_pgd(
                model,
                x,
                y,
                eps=eps,
                step_size=step_size,
                steps=steps,
                restarts=restarts,
            )
        else:
            raise ValueError(attack_name)

        if attack_name != "clean":
            c = validate_constraints(x, x_adv, eps)
            if not (c["box_ok"] and c["eps_ok"]):
                viol += 1

        with torch.no_grad():
            logits = model(x_adv)
            probs = F.softmax(logits, dim=1)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
            probs_all.append(probs.cpu())
            y_all.append(y.cpu())

    probs_cat = torch.cat(probs_all)
    y_cat = torch.cat(y_all)
    cal = calibration_bundle(probs_cat, y_cat, n_bins=15)
    return {
        "attack": attack_name,
        "eps": eps,
        "robust_acc": 100.0 * correct / max(total, 1),
        "n": total,
        "constraint_violating_batches": viol,
        "calibration": cal,
    }


def full_eval_suite(
    model: nn.Module,
    loader,
    device,
    primary_eps: float = 8 / 255,
    sensitivity_eps=(4 / 255, 8 / 255, 12 / 255),
    mr_steps: int = 50,
    mr_restarts: int = 10,
    max_batches_mr: int | None = None,
) -> dict[str, Any]:
    clean_acc, clean_probs, clean_y = _collect_clean(model, loader, device)
    out: dict[str, Any] = {
        "clean_acc": clean_acc,
        "clean_calibration": calibration_bundle(clean_probs, clean_y, n_bins=15),
        "by_eps": {},
    }
    for eps in sensitivity_eps:
        fgsm_r = evaluate_attack_loader(model, loader, device, "fgsm", eps)
        pgd_r = evaluate_attack_loader(
            model, loader, device, "pgd20", eps, steps=20, step_size=2 / 255
        )
        # Multi-restart only at primary eps by default (cost control); still required there
        entry = {"fgsm": fgsm_r, "pgd20": pgd_r}
        if abs(eps - primary_eps) < 1e-12:
            mr = evaluate_attack_loader(
                model,
                loader,
                device,
                "mr_pgd",
                eps,
                steps=mr_steps,
                restarts=mr_restarts,
                step_size=2 / 255,
                max_batches=max_batches_mr,
            )
            entry["mr_pgd"] = mr
            entry["attack_gap"] = pgd_r["robust_acc"] - mr["robust_acc"]
        out["by_eps"][str(eps)] = entry
    return out
