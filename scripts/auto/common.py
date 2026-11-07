"""Bootstrap for auto phase scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RIDOY = Path("/home/sda/Users/temp_user/.conda/envs/ridoy/bin/python")
MONITOR = ROOT / "project_monitor"
DATA_ROOT = ROOT / "dataset" / "raw"
SPLITS = ROOT / "dataset" / "splits" / "cifar10_45k5k_seed42.json"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
