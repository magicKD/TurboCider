#!/usr/bin/env python3
"""Synthetic branch attribution; not a Stage-2 bound or generated-video result."""
import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

from benchmark_ltx_qkv_replay import digest

VARIANTS = ["fused_dense", "split_dense", "direct_topk32", "pooled_topk32", "all_exact",
            "pooled_topk32_fused_post", "all_exact_fused_post"]

def validate(data, rows, runs):
    if (data.get("schema") != "ltx-attention-branch-probe-v2" or
            data.get("rows") != rows or data.get("runs") != runs or
            data.get("heads") != 32 or data.get("dim") != 128 or
            data.get("synthetic") is not True or data.get("warmups") != 2):
        raise ValueError("unexpected probe schema or shape")
    variants = data.get("variants", [])
    expected = VARIANTS
    if [v.get("name") for v in variants] != expected:
        raise ValueError("missing or reordered branch variants")
    for row in variants:
        times = [row["total_seconds"], *row["samples"]]
        if row["name"] != "fused_dense":
            times += [row[k] for k in ("qkv_seconds", "gate_seconds", "core_seconds", "post_seconds")]
        if len(row["samples"]) != runs or any(not math.isfinite(x) or x <= 0 for x in times):
            raise ValueError("invalid timing sample")
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in
               ("relative_l2_vs_fused", "max_abs_vs_fused")):
            raise ValueError("invalid numerical diagnostic")
        if row["name"].endswith("_fused_post"):
            if any(not math.isfinite(row[k]) or row[k] < 0 for k in
                   ("relative_l2_vs_unfused_post", "max_abs_vs_unfused_post")):
                raise ValueError("invalid paired numerical diagnostic")
            changed = row["changed_bf16_elements_vs_unfused_post"]
            if type(changed) is not int or not 0 <= changed <= rows * 4096:
                raise ValueError("invalid paired element count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[65, 5376, 14080])
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= args.runs <= 100 or
            not 1 <= args.rounds <= 10 or any(not 1 <= n <= 16384 for n in args.rows)):
        parser.error("use a new output, rows 1...16384, runs 1...100, rounds 1...10")
    root = Path(__file__).resolve().parents[2]
    probe = root / "build/native/ltx-attention-branch-probe"
    shader = root / "native/models/ltx_runtime/ltx_shaders.metal"
    source = root / "tools/native/ltx_attention_branch_probe.c"
    result = dict(schema="ltx-attention-branch-experiment-v2", complete=False,
        scope=__doc__, platform=platform.platform(), probe_sha256=digest(probe),
        shader_sha256=digest(shader), source_sha256=digest(source),
        gpu_host_source_sha256=digest(root / "native/models/ltx_runtime/ltx_gpu.m"),
        runner_sha256=digest(Path(__file__)),
        method="two warmups, rotating variant order per iteration; separate repeated processes",
        tokens_per_frame=0, batch_commands=False, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    for repeat in range(args.rounds):
        for n in args.rows if repeat % 2 == 0 else reversed(args.rows):
            command = [str(probe), str(shader), str(n), str(args.runs)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                result["failure"] = dict(command=command, stdout=completed.stdout,
                                         stderr=completed.stderr, returncode=completed.returncode)
                save()
                raise RuntimeError(completed.stderr)
            data = json.loads(completed.stdout)
            validate(data, n, args.runs)
            result["rows"].append(dict(repeat=repeat, command=command, measurements=data))
            save()
            print(json.dumps(dict(repeat=repeat, rows=n, variants=data["variants"])), flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
