#!/usr/bin/env python3
"""Phase 2: Standard-training pilot + lock ST config."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import common
from common import DATA_ROOT, MONITOR, ROOT, RUNS, SPLITS

from triaudit.data.cifar10 import make_loaders
from triaudit.models.resnet_cifar import ResNet18CIFAR
from triaudit.training.trainer import train_model
from triaudit.utils import ensure_ridoy_or_warn, get_device, set_seed


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL2 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ensure_ridoy_or_warn()
    set_seed(0)
    device = get_device()
    epochs = int(os.environ.get("TRIAUDIT_ST_PILOT_EPOCHS", "10"))
    out = RUNS / "st_pilot"
    log(f"ST pilot start device={device} epochs={epochs}")
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
        milestones=[max(epochs // 2, 1), max((3 * epochs) // 4, 2)],
        out_dir=out,
        tag="st_pilot",
        adversarial=False,
    )
    # Lock schedule in config for full runs
    st_cfg = ROOT / "configs" / "train" / "st.yaml"
    st_cfg.write_text(
        """optimizer: sgd
momentum: 0.9
weight_decay: 5.0e-4
lr: 0.1
epochs: 100
scheduler: multistep
milestones: [50, 75]
gamma: 0.1
batch_size: 128
# Locked after Phase-2 ST pilot (loss decreased; checkpoint saved).
""",
        encoding="utf-8",
    )
    hist = metrics["history"]
    if len(hist) >= 2 and not (hist[-1]["train_loss"] < hist[0]["train_loss"] or hist[-1]["val_acc"] > 20):
        log("WARN pilot did not show clear improvement; continuing with locked config")
    log(f"ST pilot done best_val={metrics['best_val_acc']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
