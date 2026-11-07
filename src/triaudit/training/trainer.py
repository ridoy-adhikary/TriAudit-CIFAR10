"""Shared ST / PGD-AT training loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from triaudit.attacks.pgd import pgd


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader, device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y).item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
    return {
        "acc": 100.0 * correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "n": total,
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    milestones: list[int] | None = None,
    gamma: float = 0.1,
    adversarial: bool = False,
    at_eps: float = 8 / 255,
    at_step: float = 2 / 255,
    at_steps: int = 10,
    out_dir: Path,
    tag: str = "run",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = MultiStepLR(opt, milestones=milestones or [50, 75], gamma=gamma)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_val = -1.0
    best_path = out_dir / "checkpoint.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if adversarial:
                model.eval()
                x_in = pgd(
                    model,
                    x,
                    y,
                    eps=at_eps,
                    step_size=at_step,
                    steps=at_steps,
                    random_start=True,
                )
                model.train()
            else:
                x_in = x
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_in), y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            n += y.size(0)
        sched.step()
        train_loss = running / max(n, 1)
        val = evaluate_clean(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val["acc"],
            "val_loss": val["loss"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[{tag}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_acc={val['acc']:.2f}",
            flush=True,
        )
        if val["acc"] >= best_val:
            best_val = val["acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val,
                    "tag": tag,
                },
                best_path,
            )

    metrics = {
        "tag": tag,
        "epochs": epochs,
        "best_val_acc": best_val,
        "adversarial": adversarial,
        "history": history,
        "checkpoint": str(best_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
