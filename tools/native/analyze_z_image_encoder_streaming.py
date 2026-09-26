#!/usr/bin/env python3
"""CPU-only encoder ABBA analysis. Requires NumPy; PNG analysis also needs Pillow."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct

import numpy as np


def compare_arrays(reference, candidate):
    x, y = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if x.shape != y.shape or not x.size:
        return {"valid": False, "reason": "shape_mismatch_or_empty"}
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return {"valid": False, "reason": "nonfinite"}
    x, y = x.ravel(), y.ravel()
    norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)
    identical = bool(np.array_equal(x, y))
    difference = x - y
    cosine = float(np.dot(x, y) / (norm_x * norm_y)) if norm_x and norm_y else float(identical)
    cx, cy = x - x.mean(), y - y.mean()
    centered_norm = np.linalg.norm(cx) * np.linalg.norm(cy)
    correlation = float(np.dot(cx, cy) / centered_norm) if centered_norm else float(identical)
    relative_l2 = float(np.linalg.norm(difference) / norm_x) if norm_x else (0. if identical else None)
    return {"valid": True, "identical": identical, "cosine": cosine, "correlation": correlation,
            "mae": float(np.abs(difference).mean()), "max_abs": float(np.abs(difference).max()),
            "relative_l2": relative_l2}


def read_tensor(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("oversized tensor header")
        header = json.loads(stream.read(size))["tensor"]
        raw = stream.read()
    start, end = header["data_offsets"]
    if not 0 <= start <= end <= len(raw):
        raise ValueError("tensor offsets outside payload")
    data = raw[start:end]
    dtype = header["dtype"]
    if dtype == "BF16":
        array = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        array = np.frombuffer(data, dtype={"F32": "<f4", "F16": "<f2"}[dtype])
    return array.reshape(header["shape"])


def quality_passes(rgb, initial, final, thresholds):
    if not all(value.get("valid") for value in (rgb, initial, final)):
        return False
    return (initial["identical"] and rgb["correlation"] >= thresholds["rgb_correlation_min"]
            and rgb["cosine"] >= thresholds["rgb_cosine_min"] and rgb["mae"] <= thresholds["rgb_mae_max"]
            and final["relative_l2"] is not None
            and final["relative_l2"] <= thresholds["final_latent_relative_l2_max"]
            and final["cosine"] >= thresholds["final_latent_cosine_min"])


def analyze(root):
    from PIL import Image
    plan = json.loads((root / "plan.json").read_text())
    cases = json.loads((root / "results.json").read_text())["cases"]
    if len(cases) != 4 or [c["route"] for c in cases] != plan["order"] or any(c["status"] or c["returncode"] for c in cases):
        raise ValueError("complete successful frozen ABBA run required")
    for index, case in enumerate(cases):
        if case["index"] != index:
            raise ValueError("case ordering does not match frozen plan")
        directory = root / f"{index:02d}-{case['route']}"
        if hashlib.sha256((directory / "image.png").read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError("image artifact changed after measurement")
        value = case["result"]
        if (value["model"] != plan["request"]["model"] or value["width"] != 256 or value["height"] != 256
                or value["steps"] != 9 or value["seed"] != 42 or value["prompt_cache_hit"]):
            raise ValueError("observed request differs from frozen workload")
        if case["route"] == "hybrid":
            metrics = value.get("encoder_hybrid", {})
            if metrics.get("runtime_calls_session_total") != 35 or metrics.get("runtime_failures_session_total") != 0:
                raise ValueError("incomplete or failed encoder split")
    pairs = []
    for baseline, candidate in [(0, 1), (3, 2)]:
        gpu = root / f"{baseline:02d}-gpu"
        hybrid = root / f"{candidate:02d}-hybrid"
        for field in ("pinned_blocks", "streamed_blocks", "refill_slots", "active_blocks", "memory_budget_bytes"):
            if cases[baseline]["result"]["block_streaming"][field] != cases[candidate]["result"]["block_streaming"][field]:
                raise ValueError("denoiser layouts differ between encoder routes")
        with Image.open(gpu / "image.png") as image:
            x = np.asarray(image.convert("RGB"))
        with Image.open(hybrid / "image.png") as image:
            y = np.asarray(image.convert("RGB"))
        rgb = compare_arrays(x, y)
        names = {p.name for p in (gpu / "tensors").glob("*.safetensors")}
        if names != {p.name for p in (hybrid / "tensors").glob("*.safetensors")}:
            raise ValueError("tensor diagnostic sets differ")
        tensors = {name: compare_arrays(read_tensor(gpu / "tensors" / name), read_tensor(hybrid / "tensors" / name))
                   for name in sorted(names)}
        passed = quality_passes(rgb, tensors["z_latent_initial.safetensors"], tensors["z_latent_final.safetensors"], plan["quality_thresholds"])
        passed = passed and all(value["valid"] for value in tensors.values())
        pairs.append({"gpu_index": baseline, "hybrid_index": candidate, "rgb": rgb, "tensors": tensors,
                      "quality_within_exploratory_thresholds": passed,
                      "request_wall_ratio_hybrid_over_gpu": cases[candidate]["request_wall_seconds"] / cases[baseline]["request_wall_seconds"]})
    gpu_wall = statistics.median(c["request_wall_seconds"] for c in cases if c["route"] == "gpu")
    hybrid_wall = statistics.median(c["request_wall_seconds"] for c in cases if c["route"] == "hybrid")
    gpu_process = statistics.median(c["process_wall_seconds"] for c in cases if c["route"] == "gpu")
    hybrid_process = statistics.median(c["process_wall_seconds"] for c in cases if c["route"] == "hybrid")
    return {"scope": plan["scope"], "pairs": pairs, "gpu_request_wall_median_seconds": gpu_wall,
            "hybrid_request_wall_median_seconds": hybrid_wall, "median_ratio_hybrid_over_gpu": hybrid_wall / gpu_wall,
            "gpu_worker_wait_median_seconds": gpu_process, "hybrid_worker_wait_median_seconds": hybrid_process,
            "worker_wait_ratio_hybrid_over_gpu": hybrid_process / gpu_process,
            "worker_wait_scope": "Parent wait from Popen return to worker exit; excludes Popen call itself",
            "gpu_repeat_image_identical": cases[0]["image_sha256"] == cases[3]["image_sha256"],
            "hybrid_repeat_image_identical": cases[1]["image_sha256"] == cases[2]["image_sha256"],
            "qualification": "NOT_PERFORMED: two exploratory pairs, diagnostic dumps, uncontrolled OS/Core ML caches"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))
