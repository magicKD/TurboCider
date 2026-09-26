"""Summarize matched LTX worker on/off reports; no GPU work or file writes.

Ratios are paired observations, not confidence intervals. These reports do
not themselves record the worker flag or binary identity: the launch log must
establish those independently. Exact latent equality checks scheduling parity,
not quality relative to the dense checkpoint route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TENSORS = (
    "stage1_video.bf16", "stage1_audio.bf16", "stage2_input_video.bf16",
    "stage2_video.bf16", "stage2_audio.bf16",
)


def digest(path: Path) -> dict:
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"empty tensor/report: {path}")
    return {"bytes": size, "sha256": value}


def compare(off_path: Path, on_path: Path) -> dict:
    off, on = (json.loads(path.read_text()) for path in (off_path, on_path))
    workload_off, workload_on = dict(off["workload"]), dict(on["workload"])
    flag_off = workload_off.pop("ane_persistent_worker", None)
    flag_on = workload_on.pop("ane_persistent_worker", None)
    flags_recorded = flag_off is not None or flag_on is not None
    if flags_recorded and (flag_off is not False or flag_on is not True):
        raise ValueError("reports do not record the expected worker flags")
    if workload_off != workload_on:
        raise ValueError("workloads differ")
    rows_off = off["modes"]["resident"]["runs"]
    rows_on = on["modes"]["resident"]["runs"]
    if not rows_off or len(rows_off) != len(rows_on):
        raise ValueError("missing or unmatched runs")
    rows = []
    for index, (left, right) in enumerate(zip(rows_off, rows_on)):
        if left["run"] != index or right["run"] != index:
            raise ValueError("run indices differ or are not consecutive")
        if left["result"]["ane_profile"] != right["result"]["ane_profile"]:
            raise ValueError("ANE profiles differ")
        timings = {}
        for key in ("stage1", "stage2", "request_wall"):
            a = float(left["result"]["timings_seconds"][key])
            b = float(right["result"]["timings_seconds"][key])
            if not (0 < a < float("inf") and 0 < b < float("inf")):
                raise ValueError("invalid timing")
            timings[key] = {"off_seconds": a, "on_seconds": b, "off_over_on": a / b}
        tensor_rows = {}
        for name in TENSORS:
            a, b = (digest(Path(row["stage2_video"]).parent / name)
                    for row in (left, right))
            tensor_rows[name] = {"off": a, "on": b, "byte_exact": a == b}
        rows.append({"run": index, "timings": timings, "tensors": tensor_rows})
    return {
        "schema": "ltx-worker-ablation-comparison-v1",
        "reports": {str(path): digest(path) for path in (off_path, on_path)},
        "workload": workload_off,
        "worker_flags_recorded": flags_recorded,
        "rows": rows,
        "all_tensors_byte_exact": all(t["byte_exact"] for r in rows
                                      for t in r["tensors"].values()),
        "limitations": [
            "Launch provenance must independently establish worker flags and identical binary.",
            "Sequential off/on processes do not exclude order or external resource interference.",
            "Latent equality is scheduling parity, not dense-relative perceptual qualification.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.off, args.on), indent=2, allow_nan=False))
