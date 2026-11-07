"""Shared helpers for TriAudit runs."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_ridoy_or_warn() -> str:
    exe = str(Path(__import__("sys").executable).resolve())
    if "/.conda/envs/ridoy/" not in exe:
        raise RuntimeError(f"Refusing non-ridoy python: {exe}")
    return exe


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_yaml_simple(path: Path) -> dict:
    """Minimal YAML subset loader (key: value) to avoid extra deps."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.lower() in {"true", "false"}:
                data[k] = v.lower() == "true"
            else:
                try:
                    data[k] = json.loads(v)
                except Exception:
                    data[k] = v
        return data


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))
