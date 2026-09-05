#!/usr/bin/env python3
"""Standalone entry point for ``turbocider prepare-model``."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turbocider.cli import main


if __name__ == "__main__":
    main(["prepare-model", *sys.argv[1:]])
