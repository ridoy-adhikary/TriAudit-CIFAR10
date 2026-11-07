#!/usr/bin/env python3
"""Phase 4: paired ST + PGD-AT training for seeds 0,1,2; freeze checkpoints."""

from __future__ import annotations

import os
from datetime import datetime

import torch

import common
from common import DATA_ROOT, MONITOR, RUNS, SPLITS

from triaudit.data.cifar10 import make_loaders
from triaudit.models.resnet_cifar import ResNet18CIFAR
from triaudit.training.trainer import train_model
from triaudit.utils import ensure_ridoy_or_warn, get_device, save_json, set_seed


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL4 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ensure_ridoy_or_warn()
    device = get_device()
    seeds = [0, 1, 2]
    st_epochs = int(os.environ.get("TRIAUDIT_ST_EPOCHS", "100"))
    at_epochs = int(os.environ.get("TRIAUDIT_AT_EPOCHS", "50"))
    train_loader, val_loader, _ = make_loaders(
        str(DATA_ROOT), str(SPLITS), batch_size=128, num_workers=4
    )

    summary = {"seeds": seeds, "st_epochs": st_epochs, "at_epochs": at_epochs, "runs": {}}
    for seed in seeds:
        log(f"=== seed {seed}: shared init + ST ===")
        set_seed(seed)
        init_model = ResNet18CIFAR()
        init_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}

        # ST
        set_seed(seed)
        st_model = ResNet18CIFAR()
        st_model.load_state_dict(init_state)
        st_dir = RUNS / f"seed{seed}" / "st"
        st_metrics = train_model(
            st_model,
            train_loader,
            val_loader,
            device,
            epochs=st_epochs,
            lr=0.1,
            milestones=[50, 75] if st_epochs >= 100 else [st_epochs // 2, (3 * st_epochs) // 4],
            out_dir=st_dir,
            tag=f"st_seed{seed}",
            adversarial=False,
        )

        # PGD-AT from same init
        log(f"=== seed {seed}: PGD-AT ===")
        set_seed(seed)
        at_model = ResNet18CIFAR()
        at_model.load_state_dict(init_state)
        at_dir = RUNS / f"seed{seed}" / "pgd_at"
        at_metrics = train_model(
            at_model,
            train_loader,
            val_loader,
            device,
            epochs=at_epochs,
            lr=0.1,
            milestones=[25, 40] if at_epochs >= 50 else [max(at_epochs // 2, 1), max((3 * at_epochs) // 4, 2)],
            out_dir=at_dir,
            tag=f"at_seed{seed}",
            adversarial=True,
            at_eps=8 / 255,
            at_step=2 / 255,
            at_steps=10,
        )
        # freeze marker
        for d in (st_dir, at_dir):
            (d / "FROZEN").write_text("frozen after phase4\n", encoding="utf-8")
        summary["runs"][str(seed)] = {
            "st": st_metrics["checkpoint"],
            "st_best_val": st_metrics["best_val_acc"],
            "at": at_metrics["checkpoint"],
            "at_best_val": at_metrics["best_val_acc"],
        }
        # free GPU
        del st_model, at_model
        torch.cuda.empty_cache()

    save_json(RUNS / "paired_training_summary.json", summary)
    log("paired training complete for seeds 0,1,2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
