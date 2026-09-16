#!/usr/bin/env python3
"""Reference-flow temporal diagnostics; descriptive, not a calibrated quality gate.

Compare RGB temporal residuals along dense-reference motion. The same reference
flow and reference-only forward/backward consistency mask are used for both
videos, so candidate artifacts cannot select their own favorable pixels.
Farneback flow can fail on fast motion, occlusions and textureless regions.
This does not qualify identity, semantics, perceptual fidelity or lip-sync.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from benchmark_ltx_qkv_replay import digest
from video_quality_gate import _command, _probe, _decode_process, _read_frame


FLOW_PARAMETERS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                       poly_n=5, poly_sigma=1.2, flags=0)


def flow(first, second):
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(first, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), None, **FLOW_PARAMETERS)


def temporal_metrics(a, b, c, d, forward, backward, candidate_forward):
    """a,b are reference t,t+1; c,d candidate; flow is t -> t+1 in pixels."""
    if (a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 2 or
            any(x.shape != a.shape or x.dtype != np.uint8 for x in (a, b, c, d))):
        raise ValueError("expected equally shaped uint8 RGB frames")
    h, w = a.shape[:2]
    if any(x.shape != (h, w, 2) or x.dtype != np.float32 or
           not np.isfinite(x).all() for x in (forward, backward, candidate_forward)):
        raise ValueError("invalid flow")
    y, x = np.mgrid[:h, :w].astype(np.float32)
    mx, my = x + forward[..., 0], y + forward[..., 1]
    # Keep the whole bilinear footprint inside the image.
    valid = (mx >= 0) & (my >= 0) & (mx < w - 1) & (my < h - 1)
    def warp(value):
        return cv2.remap(value, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    reverse = warp(backward)
    fb_error = np.sum((forward + reverse) ** 2, axis=-1)
    magnitude = np.sum(forward ** 2 + reverse ** 2, axis=-1)
    # This fixed diagnostic mask is not an occlusion-ground-truth claim.
    valid &= fb_error <= 0.01 * magnitude + 0.5
    count = int(valid.sum())
    if count == 0:
        raise ValueError("no flow-consistent in-bounds pixels")
    reference = warp(b.astype(np.float32)) - a.astype(np.float32)
    candidate = warp(d.astype(np.float32)) - c.astype(np.float32)
    delta = (candidate - reference)[valid]
    return dict(valid_pixels=count, pixels=h * w, valid_fraction=count / (h * w),
                reference_warp_mae_255=float(np.abs(reference[valid]).mean()),
                candidate_reference_flow_warp_mae_255=float(np.abs(candidate[valid]).mean()),
                residual_difference_mae_255=float(np.abs(delta).mean()),
                residual_difference_rmse_255=float(np.sqrt(np.mean(delta ** 2))),
                flow_endpoint_difference_px=float(np.linalg.norm(
                    candidate_forward[valid] - forward[valid], axis=-1).mean()))


def compare(reference, candidate, expected_frames, expected_fps, max_width):
    ffmpeg, ffprobe = _command("ffmpeg", None), _command("ffprobe", None)
    left, right = _probe(reference, ffprobe), _probe(candidate, ffprobe)
    if (left["width"], left["height"]) != (right["width"], right["height"]):
        raise ValueError("video dimensions differ")
    if any(not math.isfinite(v["fps"]) or v["fps"] <= 0 or
           abs(v["fps"] - expected_fps) > 1e-6 for v in (left, right)):
        raise ValueError("video fps differs from workload")
    width, height = left["width"], left["height"]
    factor = min(1., max_width / width)
    size = (max(2, round(width * factor)), max(2, round(height * factor)))
    frame_bytes = width * height * 3
    processes = [_decode_process(path, ffmpeg) for path in (reference, candidate)]
    previous = None
    rows, count = [], 0
    try:
        while True:
            payloads = [_read_frame(p.stdout, frame_bytes, path)
                        for p, path in zip(processes, (reference, candidate))]
            if all(p is None for p in payloads):
                break
            if any(p is None for p in payloads):
                raise ValueError("decoded frame counts differ")
            frames = [np.frombuffer(p, np.uint8).reshape(height, width, 3) for p in payloads]
            if size != (width, height):
                frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
            count += 1
            if previous is not None:
                a, c = previous
                b, d = frames
                rows.append(dict(frame=count - 1, **temporal_metrics(
                    a, b, c, d, flow(a, b), flow(b, a), flow(c, d))))
            previous = frames
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
    if count != expected_frames or len(rows) != count - 1 or not rows:
        raise ValueError("decoded length differs from workload or lacks transitions")
    fields = [k for k in rows[0] if k not in ("frame", "pixels", "valid_pixels")]
    # MAEs are valid-pixel weighted. RMSE is pooled via squared errors.
    weights = np.array([r["valid_pixels"] for r in rows], dtype=np.float64)
    summary = {}
    for key in fields:
        values = np.array([r[key] for r in rows])
        weighted = (np.sqrt(np.average(values ** 2, weights=weights))
                    if "rmse" in key else np.average(values, weights=weights))
        summary[key] = dict(weighted_mean=float(weighted),
                            median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                            maximum=float(values.max()), minimum=float(values.min()))
    summary["valid_fraction"]["weighted_mean"] = float(weights.sum() / sum(r["pixels"] for r in rows))
    return dict(reference=left, candidate=right, frames=count, transitions=len(rows),
                evaluation_width=size[0], evaluation_height=size[1],
                summary=summary, per_transition=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=640)
    args = parser.parse_args()
    if args.output.exists() or not 32 <= args.max_width <= 8192:
        parser.error("use a new output and max-width in 32...8192")
    pairs = []
    for path in args.report:
        data = json.loads(path.read_text())
        if data.get("schema") != "ltx-sparse-stage2-paired-v1" or not data.get("complete"):
            parser.error("source report is not a completed paired generation")
        references = [r for r in data["runs"] if r["variant"] == "baseline"]
        candidates = [r for r in data["runs"] if r["variant"] == "candidate"]
        if not references or not candidates:
            parser.error("source report lacks baseline/candidate")
        for candidate in candidates:
            pairs.append((path, data, Path(references[0]["request"]["output"]),
                          Path(candidate["request"]["output"])))
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    result = dict(schema="ltx-video-reference-flow-v1", complete=False,
                  scope=__doc__, source_sha256=digest(Path(__file__)),
                  opencv_version=cv2.__version__, numpy_version=np.__version__,
                  flow_parameters=FLOW_PARAMETERS, max_width=args.max_width,
                  device="cpu", threads=1, resize="INTER_AREA; preserve aspect ratio",
                  mask="reference-only in-bounds and squared FB error <= 0.01*magnitude + 0.5",
                  quality_gate=None, comparisons=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    for path, data, reference, candidate in pairs:
        metrics = compare(reference, candidate, data["workload"]["frames"],
                          data["workload"]["fps"], args.max_width)
        result["comparisons"].append(dict(source_report=str(path.resolve()),
            source_report_sha256=digest(path), workload=data["workload"],
            candidate=data["candidate"], reference_video=str(reference),
            reference_sha256=digest(reference), candidate_video=str(candidate),
            candidate_sha256=digest(candidate), metrics=metrics))
        save()
        print(json.dumps(dict(report=str(path), summary=metrics["summary"])), flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
