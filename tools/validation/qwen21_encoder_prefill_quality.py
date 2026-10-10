#!/usr/bin/env python3
"""CPU-only complete tensor/PNG comparison of matched prefill dump screens."""
import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from runtime_ane_common import sha256_file


def load_tensor(path):
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("missing safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= 1 << 20:
            raise ValueError("unbounded safetensors header")
        header = json.loads(source.read(size))
        payload = source.read()
    fields = {k: v for k, v in header.items() if k != "__metadata__"}
    if set(fields) != {"tensor"}:
        raise ValueError("one complete native dump tensor required")
    value = fields["tensor"]
    shape = value["shape"]
    dtype = value["dtype"]
    if (dtype not in ("F16", "BF16", "F32") or not shape or
        any(type(d) is not int or d <= 0 for d in shape)):
        raise ValueError("unsupported actual tensor geometry/dtype")
    width = 4 if dtype == "F32" else 2
    if value["data_offsets"] != [0, math.prod(shape) * width] or len(payload) != math.prod(shape) * width:
        raise ValueError("incomplete/misaligned dump payload")
    if dtype == "BF16":
        array = (np.frombuffer(payload, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        array = np.frombuffer(payload, dtype="<f4" if dtype == "F32" else "<f2")
    array = array.astype(np.float64).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite native tensor")
    return array


def difference(baseline, optimized):
    if baseline.shape != optimized.shape:
        raise ValueError("comparison did not consume the same complete tensor")
    delta = optimized - baseline
    return dict(shape=list(baseline.shape), relative_l2=float(np.linalg.norm(delta.ravel()) /
                max(np.linalg.norm(baseline.ravel()), 1e-20)),
                rmse=float(np.sqrt(np.mean(delta * delta))), max_abs=float(np.max(np.abs(delta))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-only", action="store_true", help="whole image diagnostic for matching no-dump performance screens")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh quality evidence file required")
    roots = [args.baseline.resolve(strict=True), args.optimized.resolve(strict=True)]
    summaries = [json.loads((p / "summary.json").read_text()) for p in roots]
    a, b = summaries
    if (any(s["status"] != "complete_diagnostic" or s.get("dump_tensors") is not (not args.png_only) for s in summaries) or
        a.get("encoder_prefill") is not False or b.get("encoder_prefill") is not True or
        any(a[k] != b[k] for k in ("steps", "phase", "shared_down", "decode_workers")) or
        any(b["source_identities"].get(k) != v for k, v in a["source_identities"].items())):
        raise ValueError("original runtime/source/protocol is not matched")
    if set(a["order"]) != set(b["order"]):
        raise ValueError("GPU/ANE arms differ")
    rows = []
    for mode in a["order"]:
        for index in range(3):
            row = dict(mode=mode, index=index, tensors={})
            for name in (() if args.png_only else ("text", "initial", "latents", "pixels")):
                paths = [p / f"{mode}-{index}-dump/qwen21_{name}.safetensors" for p in roots]
                hashes = [sha256_file(p) for p in paths]
                metrics = difference(*(load_tensor(p) for p in paths))
                if name == "initial" and (hashes[0] != hashes[1] or metrics["max_abs"] != 0):
                    raise ValueError("same seed did not produce the same complete native initial noise")
                if name == "text" and metrics["relative_l2"] >= .08:
                    raise ValueError("conditioning exceeds explicit8% screening budget")
                row["tensors"][name] = dict(**metrics, baseline_sha256=hashes[0], optimized_sha256=hashes[1])
            paths = [p / f"{mode}-{index}.png" for p in roots]
            images = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255 for p in paths]
            row["png"] = dict(**difference(*images), baseline_sha256=sha256_file(paths[0]), optimized_sha256=sha256_file(paths[1]))
            rows.append(row)
    report = dict(schema="tc-qwen21-encoder-prefill-quality-v1", status="complete_diagnostic", qualification_passed=False,
                  steps=a["steps"], samples=rows,
                  summaries={str(p / "summary.json"): sha256_file(p / "summary.json") for p in roots},
                  png_only=args.png_only,
                  scope=("all six matched native request pairs, complete RGB PNGs only; no tensor/semantic quality/physical placement certification" if args.png_only else
                         "all six matched native request pairs, complete tensors/RGB PNGs; identical initial noise, hidden8% screen; not semantic quality/physical placement certification"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tensor_pairs": sum(len(r["tensors"]) for r in rows),
        "max_png_relative_l2": max(r["png"]["relative_l2"] for r in rows),
        "max_png_rmse": max(r["png"]["rmse"] for r in rows),
        "max_relative_l2": {name: max(r["tensors"][name]["relative_l2"] for r in rows)
                            for name in (() if args.png_only else ("text", "initial", "latents", "pixels"))}}))


if __name__ == "__main__":
    main()
