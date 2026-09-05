#!/usr/bin/env python3
"""Run a TurboCider direct/integrated benchmark spec."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turbocider.benchmark import main


if __name__ == "__main__":
    main()
