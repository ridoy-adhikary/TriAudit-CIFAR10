#!/usr/bin/env python3
"""Thin wrapper: rebuild CIFAR split via Phase-1 impl helper."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "auto" / "impl_phase1.py"), run_name="__prepare_only__")
