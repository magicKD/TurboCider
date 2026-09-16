#!/usr/bin/env python3
"""Reproducible native sparse-core parity and performance matrix.

Build the probe first; GPU runs require a non-headless Metal-capable session.
This does not claim Stage-2 or decoded-video quality from random tensors.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; preserve previous measurements")
    if not 1 <= args.runs <= 100:
        parser.error("runs must be 1...100")
    root = Path(__file__).resolve().parents[2]
    binary = root / "build/native/ltx-attention-probe"
    shaders = root / "native/models/ltx_runtime/ltx_shaders.metal"
    sources = [binary, shaders, root / "native/models/ltx_runtime/ltx_gpu.m",
               root / "native/models/ltx_runtime/ltx_gpu.h",
               root / "tools/native/ltx_attention_probe.c"]
    report = {
        "schema": "ltx-sparse-native-core-v1",
        "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "scope": "random BF16 complete attention core; not Stage-2 or video quality",
        "warmup_pairs": 2,
        "timing": "alternating dense/candidate synchronous wall time; median",
        "rows": [], "complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def run(label, n, heads, tau, mode, radius, anchors, frame_tokens, keep=0,
            legacy=False):
        command = [str(binary), str(shaders), str(n), str(heads), str(args.runs),
                   str(tau), str(mode), str(radius), str(anchors), str(frame_tokens), str(keep)]
        if legacy:
            command = command[:6]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        row = json.loads(result.stdout)
        row.update(label=label, command=command)
        report["rows"].append(row)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"label": label, **row}), flush=True)
        return row

    # Non-64-aligned tail, single token, frame-crossing tiles and all-exact
    # limits. Every <=257-token run must pass the native scalar oracle.
    for n in (1, 65, 257):
        run("parity-legacy-sol", n, 2, .5, 0, 1, 0, 0, legacy=True)
        for mode in range(4):
            run("parity-tail", n, 2, .5, mode, 0, 0, 0)
    for mode in range(4):
        run("parity-frame-boundaries", 240, 2, .5, mode, 0, 3, 80)
        row = run("parity-all-exact", 257, 2, .5, mode, 256, 0, 0)
        if row["relative_l2"] > .03 or row["exact_block_fraction"] != 1:
            raise RuntimeError("all-exact sparse/dense parity failed")
    for n in (1, 65, 257):
        for keep in (1, 2, 256):
            run("parity-topk", n, 2, .5, 4, 0, 0, 0, keep)
            run("parity-pooled-topk", n, 2, .5, 5, 0, 0, 0, keep)
    if not args.parity_only:
        # Actual 121-frame output buckets have 16 latent frames:
        # 768x448 -> 24*14*16=5376; 1280x704 -> 40*22*16=14080.
        # The model config specifies 32 video heads of dimension 128.
        for n, frame_tokens in ((5376, 336), (14080, 880)):
            for tau in (0., .5, 1.):
                run("sol", n, 32, tau, 0, 1, 0, 0)
            for mode in (1, 2, 3):
                run("temporal-window-anchors", n, 32, .5, mode, 1, 16, frame_tokens)
            for keep in (16, 32, 64):
                run("direct-topk-blocks", n, 32, .5, 4, 0, 0, 0, keep)
                run("pooled-topk-blocks", n, 32, .5, 5, 0, 0, 0, keep)
    report["complete"] = True
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
