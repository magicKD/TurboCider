#!/usr/bin/env python3
"""Validate and embed the frozen Qwen 2.1 SmoothQuant profile in the native library."""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "native/models/qwen21/calibration"

def generate(directory: Path, output: Path):
    meta = json.loads((directory / "sq_convrot_v1.json").read_text())
    data = (directory / "sq_convrot_v1.f16le").read_bytes()
    if meta["schema"] != "qwen21-sq-convrot-profile-v1" or hashlib.sha256(data).hexdigest() != meta["payload_sha256"]:
        raise ValueError("SmoothQuant calibration payload integrity mismatch")
    for name in ("identity", "source_sha256", "payload_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", meta[name]):
            raise ValueError("Invalid calibration identity")
    offset = 0
    keys = []
    for entry in meta["entries"]:
        key, count = entry["key"], entry["channels"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", key) or ".." in key:
            raise ValueError("Invalid projection key")
        if entry["offset"] != offset or count <= 0 or count % 64:
            raise ValueError("Invalid calibration geometry")
        offset += count
        keys.append(key)
    if len(keys) != 392 or keys != sorted(set(keys)) or offset * 2 != len(data):
        raise ValueError("Incomplete calibration profile")
    if any(not math.isfinite(x) or not 1/16 <= x <= 16 for x, in struct.iter_unpack("<e", data)):
        raise ValueError("Invalid inverse scale")
    lines = ["#pragma once", "#include <cstdint>", "namespace tc::qwen21::sq_profile {",
             "struct Entry { const char *key; int offset, channels; };"]
    for name in ("identity", "source_sha256", "payload_sha256"):
        lines.append(f'inline constexpr const char *{name} = "{meta[name]}";')
    lines.append("inline constexpr Entry entries[] = {")
    lines.extend('{' + json.dumps(e["key"]) + f', {e["offset"]}, {e["channels"]}' + '},' for e in meta["entries"])
    lines.append("};\ninline constexpr uint16_t inverse_bits[] = {")
    values = [str(x) for x, in struct.iter_unpack("<H", data)]
    lines.extend(",".join(values[i:i+32]) + "," for i in range(0, len(values), 32))
    lines.append("};\n}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(ASSETS, args.output)
