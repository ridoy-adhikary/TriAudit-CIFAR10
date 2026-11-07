#!/usr/bin/env python3
"""Phase 5: attack evaluation on frozen checkpoints."""

from __future__ import annotations

import os
from datetime import datetime

import torch

import common
from common import DATA_ROOT, MONITOR, RUNS, SPLITS

from triaudit.data.cifar10 import make_loaders
from triaudit.evaluation.harness import full_eval_suite
from triaudit.models.resnet_cifar import ResNet18CIFAR
from triaudit.utils import ensure_ridoy_or_warn, get_device, save_json


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL5 | {msg}"
    print(line, flush=True)
    with (MONITOR / "project_activity.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_frozen(path):
    ckpt = torch.load(path, map_location="cpu")
    model = ResNet18CIFAR()
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> int:
    ensure_ridoy_or_warn()
    device = get_device()
    _, _, test_loader = make_loaders(
        str(DATA_ROOT), str(SPLITS), batch_size=128, num_workers=4
    )
    # Optional cost control for multi-restart (None = full test set)
    max_mr = os.environ.get("TRIAUDIT_MR_MAX_BATCHES")
    max_batches_mr = int(max_mr) if max_mr else None

    out = {"seeds": {}}
    for seed in (0, 1, 2):
        for cond in ("st", "pgd_at"):
            ckpt = RUNS / f"seed{seed}" / cond / "checkpoint.pt"
            log(f"eval attacks seed={seed} cond={cond}")
            model = load_frozen(ckpt).to(device)
            result = full_eval_suite(
                model,
                test_loader,
                device,
                primary_eps=8 / 255,
                mr_steps=50,
                mr_restarts=10,
                max_batches_mr=max_batches_mr,
            )
            key = f"seed{seed}_{cond}"
            out["seeds"][key] = result
            save_json(RUNS / "eval" / f"attacks_{key}.json", result)
            del model
            torch.cuda.empty_cache()

    save_json(RUNS / "eval" / "attacks_summary.json", out)
    log("attack evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
