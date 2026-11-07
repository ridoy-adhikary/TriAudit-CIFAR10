"""CIFAR-10 loaders with persisted 45k/5k split.

Images are returned in [0, 1] (ToTensor only). Normalization is applied inside the model
so adversarial attacks can clip in valid pixel space.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_split_indices(splits_file: str | Path):
    data = json.loads(Path(splits_file).read_text(encoding="utf-8"))
    return data["train_indices"], data["val_indices"]


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose([transforms.ToTensor()])


def make_loaders(
    root,
    splits_file,
    batch_size=128,
    num_workers=4,
    mean=None,
    std=None,
):
    del mean, std  # kept for config compatibility; normalization is model-side
    train_idx, val_idx = load_split_indices(splits_file)
    base_train = CIFAR10(
        root=root, train=True, download=False, transform=build_transforms(True)
    )
    base_val = CIFAR10(
        root=root, train=True, download=False, transform=build_transforms(False)
    )
    base_test = CIFAR10(
        root=root, train=False, download=False, transform=build_transforms(False)
    )
    train_loader = DataLoader(
        Subset(base_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(base_val, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        base_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
