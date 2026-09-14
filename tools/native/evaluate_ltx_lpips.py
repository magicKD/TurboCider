#!/usr/bin/env python3
"""Full-resolution, all-frame LPIPS for completed paired LTX videos.

Descriptive spatial metric only. No pass threshold, timing claim or temporal,
semantic, identity or lip-sync qualification is inferred from these scores.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark_ltx_qkv_replay import digest
from video_quality_gate import _command, _probe, _decode_process, _read_frame


def rgb_tensor(payload, width, height):
    if width <= 0 or height <= 0 or len(payload) != width * height * 3:
        raise ValueError("invalid RGB payload")
    array = np.frombuffer(payload, np.uint8).copy().reshape(height, width, 3)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.


def validate_pair(left, right):
    if (left["width"], left["height"]) != (right["width"], right["height"]):
        raise ValueError("video dimensions differ")
    if (not math.isfinite(left["fps"]) or not math.isfinite(right["fps"]) or
            min(left["fps"], right["fps"]) <= 0 or abs(left["fps"] - right["fps"]) > 1e-6):
        raise ValueError("video fps differs or is unknown")


def compare(reference, candidate, expected_frames, model):
    ffmpeg, ffprobe = _command("ffmpeg", None), _command("ffprobe", None)
    left, right = _probe(reference, ffprobe), _probe(candidate, ffprobe)
    validate_pair(left, right)
    frame_bytes = left["width"] * left["height"] * 3
    processes = [_decode_process(path, ffmpeg) for path in (reference, candidate)]
    scores, self_distance = [], None
    try:
        with torch.inference_mode():
            while True:
                a = _read_frame(processes[0].stdout, frame_bytes, reference)
                b = _read_frame(processes[1].stdout, frame_bytes, candidate)
                if a is None and b is None:
                    break
                if a is None or b is None:
                    raise ValueError("decoded frame counts differ")
                x = rgb_tensor(a, left["width"], left["height"])
                y = rgb_tensor(b, left["width"], left["height"])
                if self_distance is None:
                    self_distance = float(model(x, x).item())
                    if not math.isfinite(self_distance) or abs(self_distance) > 1e-6:
                        raise ValueError("LPIPS identity check failed")
                score = float(model(x, y).item())
                if not math.isfinite(score):
                    raise ValueError("nonfinite LPIPS")
                scores.append(score)
                if len(scores) % 30 == 0:
                    print(f"{candidate.parent.name}: {len(scores)} frames", flush=True)
        for process in processes:
            stderr = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
            process.stderr.close()
    if not scores or len(scores) != expected_frames:
        raise ValueError("decoded length differs from benchmark workload")
    return dict(reference=left, candidate=right, frames=len(scores), scores=scores,
                mean=float(np.mean(scores)), median=float(np.median(scores)),
                p95=float(np.percentile(scores, 95)), maximum=max(scores),
                identity_check=self_distance)


def report_pairs(path):
    data = json.loads(path.read_text())
    schema = data.get("schema")
    if not data.get("complete"):
        raise ValueError("source report is incomplete")
    if schema in ("ltx-sparse-stage2-paired-v1", "ltx-ane-stage2-paired-v1"):
        baseline = next((r for r in data.get("runs", []) if r.get("variant") == "baseline"), None)
        candidates = [r for r in data.get("runs", []) if r.get("variant") == "candidate"]
        if baseline is None or not candidates:
            raise ValueError("paired generation report lacks baseline or candidates")
        return [dict(reference=Path(baseline["request"]["output"]),
                     candidate_video=Path(candidate["request"]["output"]),
                     expected_frames=data["workload"]["frames"],
                     workload=data["workload"], candidate=data["candidate"],
                     isolation=None)
                for candidate in candidates]
    if schema == "ltx-vae-tiling-quality-v1":
        pairs = []
        for comparison in data.get("comparisons", []):
            videos = comparison.get("videos", [])
            tensor_checks = comparison.get("tensor_checks", {})
            if len(videos) != 2 or not tensor_checks or not all(
                    check.get("byte_exact") for check in tensor_checks.values()):
                raise ValueError("VAE tiling comparison is not isolated by exact tensors")
            metrics = comparison.get("metrics", {})
            expected_frames = metrics.get("frame_count_compared")
            if not isinstance(expected_frames, int) or expected_frames < 1:
                raise ValueError("VAE tiling comparison has no valid frame count")
            reference_meta = metrics.get("reference", {})
            pairs.append(dict(reference=Path(videos[0]), candidate_video=Path(videos[1]),
                              expected_frames=expected_frames,
                              workload=dict(operation="vae_decode", width=reference_meta.get("width"),
                                            height=reference_meta.get("height"),
                                            frames=expected_frames, fps=reference_meta.get("fps")),
                              candidate=dict(operation="vae_decode", variant="spatial_tiling",
                                             run=comparison.get("run")),
                              isolation=dict(tensor_checks=tensor_checks,
                                             video_sha256=comparison.get("video_sha256"))))
        if not pairs:
            raise ValueError("VAE tiling report has no comparisons")
        return pairs
    raise ValueError("source report schema is unsupported")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--lpips-source", type=Path, required=True)
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 32:
        parser.error("use a new output and threads in 1...32")
    pairs = []
    for path in args.report:
        try:
            pairs.extend((path.resolve(), pair) for pair in report_pairs(path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"invalid source report {path}: {error}")
    source = args.lpips_source.resolve()
    if args.deps:
        sys.path.insert(0, str(args.deps.resolve()))
    sys.path.insert(0, str(source))
    import lpips
    if not Path(lpips.__file__).resolve().is_relative_to(source):
        raise ValueError("LPIPS imported from a different source")
    torch.hub.set_dir(str(args.cache.resolve() / "hub"))
    torch.set_num_threads(args.threads)
    model = lpips.LPIPS(net="alex", version="0.1", pretrained=True, pnet_rand=False,
                        spatial=False, eval_mode=True, verbose=False).cpu().eval()
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=True).stdout.strip()
    result = dict(schema="ltx-video-lpips-v1", complete=False, scope=__doc__,
                  metric="LPIPS alex v0.1", device="cpu", threads=args.threads,
                  normalization="RGB uint8 -> float32 [-1,1], NCHW; no resize/crop",
                  lpips_commit=commit, model_state_sha256=state_hash.hexdigest(),
                  source_sha256=digest(Path(__file__)),
                  versions={name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "numpy", "scipy", "tqdm")},
                  comparisons=[], quality_gate=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    for path, pair in pairs:
        reference, candidate = pair["reference"], pair["candidate_video"]
        row = dict(source_report=str(path), source_report_sha256=digest(path),
                   reference_video=str(reference), reference_sha256=digest(reference),
                   candidate_video=str(candidate), candidate_sha256=digest(candidate),
                   workload=pair["workload"], candidate=pair["candidate"],
                   isolation=pair["isolation"],
                   metrics=compare(reference, candidate, pair["expected_frames"], model))
        result["comparisons"].append(row)
        save()
        print(json.dumps(dict(report=str(path), mean=row["metrics"]["mean"])), flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
