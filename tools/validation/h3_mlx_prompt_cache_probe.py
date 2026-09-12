#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-cache-") as directory:
        root = Path(directory)
        (root / "component").mkdir()
        (root / "tokenizer").mkdir()
        subprocess.run([
            str(args.probe), str(root / "component"), str(root / "tokenizer"), str(root / "cache")
        ], check=True)


if __name__ == "__main__":
    main()
