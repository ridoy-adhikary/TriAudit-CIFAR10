#!/usr/bin/env python3
"""Phase 3: PGD-AT pilot + constraint unit test."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch

import common
from common import DATA_ROOT, MONITOR, ROOT, RUNS, SPLITS

from triaudit.attacks.pgd import pgd, validate_constraints
from triaudit.data.cifar10 import make_loaders
from triaudit.models.resnet_cifar import ResNet18CIFAR
from triaudit.training.trainer import train_model
from triaudit.utils import ensure_ridoy_or_warn, get_device, set_seed


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL3 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_constraint_test() -> None:
    tests = ROOT / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test_pgd_constraints.py").write_text(
        '''import torch
from triaudit.attacks.pgd import pgd, validate_constraints
from triaudit.models.resnet_cifar import ResNet18CIFAR

def test_pgd_constraints():
    model = ResNet18CIFAR().eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    eps = 8/255
    adv = pgd(model, x, y, eps=eps, step_size=2/255, steps=7, random_start=True)
    c = validate_constraints(x, adv, eps)
    assert c["box_ok"] and c["eps_ok"], c
''',
        encoding="utf-8",
    )


def main() -> int:
    ensure_ridoy_or_warn()
    write_constraint_test()
    set_seed(0)
    device = get_device()
    # quick constraint smoke
    model = ResNet18CIFAR().to(device).eval()
    x = torch.rand(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    adv = pgd(model, x, y, eps=8 / 255, step_size=2 / 255, steps=10, random_start=True)
    c = validate_constraints(x, adv, 8 / 255)
    assert c["box_ok"] and c["eps_ok"], c
    log(f"constraint smoke OK max_linf={c['max_linf']:.6f}")

    epochs = int(os.environ.get("TRIAUDIT_AT_PILOT_EPOCHS", "3"))
    out = RUNS / "pgd_at_pilot"
    train_loader, val_loader, _ = make_loaders(
        str(DATA_ROOT), str(SPLITS), batch_size=128, num_workers=4
    )
    model = ResNet18CIFAR()
    metrics = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=0.1,
        milestones=[max(epochs, 1)],
        out_dir=out,
        tag="pgd_at_pilot",
        adversarial=True,
        at_eps=8 / 255,
        at_step=2 / 255,
        at_steps=10,
    )
    (ROOT / "configs" / "train" / "pgd_at.yaml").write_text(
        """optimizer: sgd
momentum: 0.9
weight_decay: 5.0e-4
lr: 0.1
epochs: 50
scheduler: multistep
milestones: [25, 40]
gamma: 0.1
batch_size: 128
attack:
  name: pgd
  steps: 10
  eps: 0.03137254901960784
  step_size: 0.00784313725490196
  random_start: true
  clip_min: 0.0
  clip_max: 1.0
# Locked after Phase-3 PGD-AT pilot. Full paired AT uses 50 epochs (AT cost).
""",
        encoding="utf-8",
    )
    log(f"PGD-AT pilot done best_val={metrics['best_val_acc']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
