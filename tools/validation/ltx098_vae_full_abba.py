#!/usr/bin/env python3
"""End-to-end GPU A/B/B/A for an isolated LTX 0.9.8 candidate.

Each process makes two 512x512x121 videos with the same prompt and consecutive
seeds. Full decode and MP4 export are timed; every decoded RGB frame is checked
against its matching baseline. No production runtime is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from ltx098_comfy_reference import MODEL_NAME, MODEL_SHA256, TEXT_NAME, TEXT_SHA256, sha256


def rgb_frame_hashes(video):
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0",
                             "-pix_fmt", "rgb24", "-f", "framemd5", "-"],
                            text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"video decode failed: {result.stderr[-4000:]}")
    hashes = [line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines()
              if line and not line.startswith("#")]
    if len(hashes) != 121 or any(len(digest) != 32 for digest in hashes):
        raise ValueError("unexpected decoded frame hashes")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", choices=("vae-padding", "rgb-cast"), default="vae-padding")
    args = parser.parse_args()
    model, image, baseline, candidate = (p.resolve(strict=True) for p in
                                          (args.model, args.input, args.baseline, args.candidate))
    output = args.output.resolve()
    if output.exists() or not args.prompt or not 0 <= args.seed < 0xffffffffffffffff or baseline == candidate:
        parser.error("new output, distinct binaries, nonempty prompt and valid uint64 seed pair required")
    checkpoint, text, tokenizer = model / MODEL_NAME, model / "text_encoders" / TEXT_NAME, model / "tokenizer/tokenizer.json"
    if sha256(checkpoint) != MODEL_SHA256 or sha256(text) != TEXT_SHA256:
        raise ValueError("checkpoint or T5 encoder differs from pinned source")
    identities = {"checkpoint_sha256": MODEL_SHA256, "text_encoder_sha256": TEXT_SHA256,
                  "tokenizer_sha256": sha256(tokenizer), "input_sha256": sha256(image),
                  "baseline_probe_sha256": sha256(baseline), "candidate_probe_sha256": sha256(candidate)}
    output.mkdir(parents=True)
    report = {"schema": "ltx098-vae-full-abba-v1", "status": "running", **identities,
              "prompt": args.prompt, "seed": args.seed, "experiment": args.experiment,
              "scope": "A/B/B/A fresh processes, each with two sequential 512x512x121 eight-step GPU requests; "
                       "first in-process wall includes initialization, second reuses prompt caches; "
                       "full decode and MP4 export included, process startup excluded", "runs": [], "quality": []}
    try:
        for label, mode, executable in (("a0", "baseline", baseline), ("b0", "candidate", candidate),
                                         ("b1", "candidate", candidate), ("a1", "baseline", baseline)):
            destination = output / label
            command = [str(executable), str(model), str(image), str(destination), "512", "121", "2", args.prompt,
                       "--seed", str(args.seed)]
            started = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            process_wall = time.perf_counter() - started
            (output / f"{label}.stdout.txt").write_text(result.stdout)
            (output / f"{label}.stderr.txt").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"{label} exited {result.returncode}: {result.stderr[-4000:]}")
            detail = json.loads((destination / "report.json").read_text())
            if detail.get("execution") != "gpu" or len(detail.get("runs", [])) != 2:
                raise ValueError("expected two GPU requests per process")
            for index, run in enumerate(detail["runs"]):
                if (run.get("index"), run.get("seed"), run.get("actual_steps"), run.get("text_cache_hit"),
                        run.get("cross_attention_cache_hit")) != (index, args.seed + index, 8, bool(index), bool(index)):
                    raise ValueError(f"{label} request {index} execution/cache contract differs")
                row = {"label": label, "mode": mode, "request_index": index, "seed": run["seed"],
                       "wall_seconds": run["wall_seconds"], "decode_seconds": run["decode_seconds"],
                       "denoise_seconds": run["denoise_seconds"], "mlx_peak_bytes": run["mlx_peak_bytes"],
                       "process_client_wall_seconds": process_wall}
                report["runs"].append(row)
                print(json.dumps(row), flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for pair in (0, 1):
            for index in (0, 1):
                first = rgb_frame_hashes(output / f"a{pair}" / f"{index}.mp4")
                second = rgb_frame_hashes(output / f"b{pair}" / f"{index}.mp4")
                same = first == second
                row = {"pair": pair, "request_index": index, "seed": args.seed + index,
                       "frames": len(first), "all_rgb_frames_identical": same,
                       "reference_frame_hash_list_sha256": hashlib.sha256("\n".join(first).encode()).hexdigest(),
                       "candidate_frame_hash_list_sha256": hashlib.sha256("\n".join(second).encode()).hexdigest()}
                report["quality"].append(row)
                print(json.dumps(row), flush=True)
        if not all(row["all_rgb_frames_identical"] for row in report["quality"]):
            raise ValueError("candidate decoded RGB frames differ from baseline")
        if (sha256(checkpoint) != identities["checkpoint_sha256"] or sha256(text) != identities["text_encoder_sha256"] or
                sha256(tokenizer) != identities["tokenizer_sha256"] or sha256(image) != identities["input_sha256"] or
                sha256(baseline) != identities["baseline_probe_sha256"] or sha256(candidate) != identities["candidate_probe_sha256"]):
            raise ValueError("source or binary identity changed during measurement")
        phases = {}
        for index, phase in enumerate(("cold", "warm")):
            selected = [row for row in report["runs"] if row["request_index"] == index]
            a = [row["wall_seconds"] for row in selected if row["mode"] == "baseline"]
            b = [row["wall_seconds"] for row in selected if row["mode"] == "candidate"]
            phases[phase] = {"baseline_median_seconds": statistics.median(a), "candidate_median_seconds": statistics.median(b),
                             "matched_speedups": [left / right for left, right in zip(a, b)],
                             "all_candidate_faster": all(right < left for left, right in zip(a, b))}
        report.update(status="complete", all_rgb_frames_identical=True, phases=phases)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["phases"], indent=2))


if __name__ == "__main__":
    main()
