#!/usr/bin/env python3
"""Pack a self-contained TurboCider Web mini app."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.source.resolve()
output = args.output.resolve()
if not (source / "manifest.json").is_file() or output.suffix != ".tcapp" or source in output.parents:
    parser.error("Use a directory containing manifest.json and an external .tcapp output path.")
files = []
for path in sorted(source.rglob("*")):
    name = path.relative_to(source)
    if any(part.startswith(".") for part in name.parts):
        continue
    if path.is_symlink():
        parser.error(f"Symlinks are unsupported: {name}")
    if path.is_file():
        files.append((path, name.as_posix()))
with ZipFile(output, "x", compression=ZIP_DEFLATED, allowZip64=False) as archive:
    for path, name in files:
        archive.write(path, arcname=name)
