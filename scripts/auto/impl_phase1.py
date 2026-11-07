#!/usr/bin/env python3
"""Phase 1 implementation: splits, package scaffold, CIFAR ResNet-18, configs."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[2]
RIDOY_HINT = "/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python"
MONITOR = ROOT / "project_monitor"
ACTIVITY = MONITOR / "project_activity.log"


def log(msg: str) -> None:
    MONITOR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMPL1 | {msg}"
    print(line, flush=True)
    with ACTIVITY.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def make_split() -> Path:
    raw = ROOT / "dataset" / "raw"
    splits = ROOT / "dataset" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    out = splits / "cifar10_45k5k_seed42.json"

    ds = CIFAR10(root=str(raw), train=True, download=False)
    n = len(ds)
    assert n == 50000, n
    rng = random.Random(42)
    idx = list(range(n))
    rng.shuffle(idx)
    train_idx = sorted(idx[:45000])
    val_idx = sorted(idx[45000:])
    assert len(train_idx) == 45000 and len(val_idx) == 5000
    assert set(train_idx).isdisjoint(val_idx)

    payload = {
        "dataset": "CIFAR-10",
        "source": "official_train",
        "seed": 42,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test": "official_test_10000_untouched",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    log(f"wrote split {out}")
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """CIFAR-adapted ResNet-18: 3x3 stem, no max-pool, 10-class head."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def scaffold() -> None:
    files = {
        ROOT / "src" / "triaudit" / "__init__.py": '__version__ = "0.1.0"\n',
        ROOT / "src" / "triaudit" / "data" / "__init__.py": "",
        ROOT / "src" / "triaudit" / "models" / "__init__.py": "from .resnet_cifar import ResNet18CIFAR\n__all__ = ['ResNet18CIFAR']\n",
        ROOT / "src" / "triaudit" / "training" / "__init__.py": "",
        ROOT / "src" / "triaudit" / "attacks" / "__init__.py": "",
        ROOT / "src" / "triaudit" / "metrics" / "__init__.py": "",
        ROOT / "src" / "triaudit" / "evaluation" / "__init__.py": "",
        ROOT / "src" / "triaudit" / "reporting" / "__init__.py": "",
        ROOT / "configs" / "data" / "cifar10.yaml": """dataset: CIFAR-10
root: dataset/raw
splits_file: dataset/splits/cifar10_45k5k_seed42.json
num_classes: 10
batch_size: 128
num_workers: 4
normalize:
  mean: [0.4914, 0.4822, 0.4465]
  std: [0.2470, 0.2435, 0.2616]
""",
        ROOT / "configs" / "model" / "resnet18_cifar.yaml": """name: resnet18_cifar
num_classes: 10
""",
        ROOT / "configs" / "train" / "st.yaml": """# Locked after ST pilot (Phase 2). Pilot starting point from architecture.
optimizer: sgd
momentum: 0.9
weight_decay: 5.0e-4
lr: 0.1
epochs: 100
scheduler: multistep
milestones: [50, 75]
gamma: 0.1
batch_size: 128
""",
        ROOT / "configs" / "train" / "pgd_at.yaml": """optimizer: sgd
momentum: 0.9
weight_decay: 5.0e-4
lr: 0.1
epochs: 100
scheduler: multistep
milestones: [50, 75]
gamma: 0.1
batch_size: 128
attack:
  name: pgd
  steps: 10
  eps: 0.03137254901960784  # 8/255
  step_size: 0.00784313725490196  # 2/255
  random_start: true
  clip_min: 0.0
  clip_max: 1.0
""",
        ROOT / "configs" / "attack" / "eval_attacks.yaml": """norm: linf
epsilons: [0.01568627450980392, 0.03137254901960784, 0.047058823529411764]  # 4/255, 8/255, 12/255
primary_eps: 0.03137254901960784
fgsm: {}
pgd20:
  steps: 20
  step_size: 0.00784313725490196
  random_start: true
multi_restart_pgd:
  steps: 50
  restarts: 10
  step_size: 0.00784313725490196
  random_start: true
autoattack:
  enabled: false
""",
        ROOT / "configs" / "calibration" / "ece15.yaml": """ece_bins: 15
binning: equal_width
metrics: [ece, nll, brier]
conditions: [clean, fgsm, pgd20]
""",
        ROOT / "configs" / "experiment" / "paired_seeds.yaml": """seeds: [0, 1, 2]
pair_st_with_at: true
shared_init: true
""",
        ROOT / "scripts" / "prepare_data.py": '''#!/usr/bin/env python3
"""Thin wrapper: rebuild CIFAR split via Phase-1 impl helper."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "auto" / "impl_phase1.py"), run_name="__prepare_only__")
''',
    }

    for path, content in files.items():
        write(path, content)

    # Always refresh model module to architecture definition
    model_path = ROOT / "src" / "triaudit" / "models" / "resnet_cifar.py"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        '''"""CIFAR-adapted ResNet-18 for TriAudit."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
''',
        encoding="utf-8",
    )

    data_mod = ROOT / "src" / "triaudit" / "data" / "cifar10.py"
    data_mod.write_text(
        '''"""CIFAR-10 loaders with persisted 45k/5k split."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

def load_split_indices(splits_file: str | Path):
    data = json.loads(Path(splits_file).read_text(encoding="utf-8"))
    return data["train_indices"], data["val_indices"]

def build_transforms(train: bool, mean, std):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def make_loaders(root, splits_file, batch_size=128, num_workers=4, mean=None, std=None):
    mean = mean or [0.4914, 0.4822, 0.4465]
    std = std or [0.2470, 0.2435, 0.2616]
    train_idx, val_idx = load_split_indices(splits_file)
    base_train = CIFAR10(root=root, train=True, download=False, transform=build_transforms(True, mean, std))
    base_val = CIFAR10(root=root, train=True, download=False, transform=build_transforms(False, mean, std))
    base_test = CIFAR10(root=root, train=False, download=False, transform=build_transforms(False, mean, std))
    train_loader = DataLoader(Subset(base_train, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(base_val, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(base_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
''',
        encoding="utf-8",
    )
    log("scaffolded src/configs/scripts")


def smoke_model() -> None:
    model = ResNet18CIFAR(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), y.shape
    log(f"model smoke OK shape={tuple(y.shape)} params={sum(p.numel() for p in model.parameters())}")


def main() -> int:
    exe = str(Path(__import__('sys').executable).resolve())
    if "/.conda/envs/ridoy/" not in exe:
        log(f"WARN expected ridoy python, got {exe}; continue only if orchestrator re-execed")
    scaffold()
    make_split()
    smoke_model()
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests").mkdir(exist_ok=True)
    log("phase1 implementation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
