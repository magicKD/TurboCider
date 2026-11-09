#!/usr/bin/env python3
"""Full-request GPU vs bias-inclusive CPU_AND_NE probe, A/B/B/A.

This does not enable the hybrid route in TurboCider. Device placement remains
unverified; strict same-encoder video quality and all call counts are recorded.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from ltx098_comfy_reference import MODEL_NAME, MODEL_SHA256, TEXT_NAME, TEXT_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--step-mask", type=int, default=255, help="bit i enables hybrid FFN in denoising step i")
    args = parser.parse_args()
    probe, model, image, manifest = (p.resolve(strict=True) for p in (args.probe, args.model, args.input, args.manifest))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if not args.prompt or not 0 <= args.seed < 0xffffffffffffffff or not 1 <= args.chunks <= 15 or not 1 <= args.step_mask <= 255:
        parser.error("prompt, uint64 seed pair, 1–15 chunks and 1–255 step mask required")
    checkpoint = model / MODEL_NAME
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from pinned source")
    text_encoder, tokenizer = model / "text_encoders" / TEXT_NAME, model / "tokenizer/tokenizer.json"
    if sha256(text_encoder) != TEXT_SHA256:
        raise ValueError("text encoder SHA-256 differs from pinned source")
    spec = json.loads(manifest.read_text())
    if (spec.get("backend"), spec.get("biases"), spec.get("rows"), spec.get("hidden"), spec.get("width")) != (
            "ltx098_experimental_bias_fp16", True, 256, 2048, 8192):
        raise ValueError("incompatible experimental bias graph")
    compiled = manifest.parent / "graph.mlmodelc"
    files = {name: digest for name, digest in spec["files"].items() if name.startswith("graph.mlmodelc/")}
    if not files or compiled.is_symlink() or not compiled.is_dir():
        raise ValueError("compiled graph receipt missing")
    observed = set()
    for path in compiled.rglob("*"):
        if path.is_symlink():
            raise ValueError("compiled graph symlinks are not supported")
        if path.is_file():
            observed.add(path.relative_to(manifest.parent).as_posix())
        elif not path.is_dir():
            raise ValueError("compiled graph contains a nonregular entry")
    if observed != set(files) or any(sha256(manifest.parent / name) != digest for name, digest in files.items()):
        raise ValueError("compiled graph receipt mismatch")
    output.mkdir(parents=True)
    report = {"schema": "ltx098-ane-hybrid-abba-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "text_encoder_sha256": TEXT_SHA256, "tokenizer_sha256": sha256(tokenizer),
              "manifest_sha256": sha256(manifest), "probe_sha256": sha256(probe), "input_sha256": sha256(image),
              "prompt": args.prompt, "chunks": args.chunks, "step_mask": args.step_mask,
              "scope": "Fresh-process full-request A GPU / B CPU_AND_NE+GPU / B / A; 512x512x121, 8 steps, same prompt/seed/encoder; plan is not runtime placement proof",
              "physical_ane_placement": "unknown", "runs": [], "quality": []}
    try:
        for label, mode, seed in (("a0", "gpu", args.seed), ("b0", "hybrid", args.seed),
                                  ("b1", "hybrid", args.seed + 1), ("a1", "gpu", args.seed + 1)):
            destination = output / label
            command = [str(probe), str(model), str(image), str(compiled), str(destination), args.prompt,
                       str(seed), mode, str(args.chunks), str(args.step_mask)]
            started = time.perf_counter()
            run = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            wall = time.perf_counter() - started
            (output / f"{label}.stdout.txt").write_text(run.stdout)
            (output / f"{label}.stderr.txt").write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(f"{label} exited {run.returncode}: {run.stderr[-4000:]}")
            detail = json.loads((destination / "report.json").read_text())
            expected_ffns = 224 if mode == "hybrid" else 0
            expected_hybrid_ffns = 28 * args.step_mask.bit_count() if mode == "hybrid" else 0
            expected_coreml = expected_hybrid_ffns * args.chunks
            if (detail.get("mode"), detail.get("seed"), detail.get("actual_steps"), detail.get("ffn_calls"), detail.get("coreml_calls")) != (
                    mode, seed, 8, expected_ffns, expected_coreml):
                raise ValueError(f"{label} did not execute expected complete request/call counts")
            if detail.get("hybrid_ffn_calls") != expected_hybrid_ffns or detail.get("hybrid_step_mask") != args.step_mask:
                raise ValueError(f"{label} hybrid step selection differs")
            if mode == "hybrid" and detail["per_block_calls"] != [8] * 28:
                raise ValueError("hybrid did not execute every block at every step")
            row = {"label": label, "mode": mode, "seed": seed, "client_wall_seconds": wall,
                   "reported_wall_seconds": detail["wall_seconds"], "denoise_seconds": detail["denoise_seconds"],
                   "graph_load_and_selftest_seconds": detail["graph_load_and_selftest_seconds"],
                   "ffn_calls": detail["ffn_calls"], "hybrid_ffn_calls": detail["hybrid_ffn_calls"], "coreml_calls": detail["coreml_calls"]}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        import torch
        from safetensors.torch import load_file
        for index in (0, 1):
            ref_path, candidate_path = output / f"a{index}/final-latent.safetensors", output / f"b{index}/final-latent.safetensors"
            reference = load_file(str(ref_path))["value"].float()
            candidate = load_file(str(candidate_path))["value"].float()
            if reference.shape != candidate.shape or not torch.isfinite(candidate).all():
                raise ValueError("invalid hybrid final latent")
            delta = reference - candidate
            quality_path = output / f"quality-{index}.json"
            compare = subprocess.run([sys.executable, str(Path(__file__).with_name("ltx098_video_comparison.py")),
                "--reference", str(output / f"a{index}/0.mp4"), "--candidate", str(output / f"b{index}/0.mp4"),
                "--output", str(quality_path)], capture_output=True, text=True, timeout=600)
            (output / f"quality-{index}.stdout.txt").write_text(compare.stdout)
            (output / f"quality-{index}.stderr.txt").write_text(compare.stderr)
            if compare.returncode:
                raise RuntimeError(compare.stderr)
            quality = json.loads(quality_path.read_text())
            report["quality"].append({"seed": args.seed + index,
                "latent_relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
                "contract_passed": quality["contract_passed"], "aligned_rgb_gate_passed": quality["aligned_rgb_gate_passed"],
                "mean_correlation": quality["metrics"]["mean_correlation"], "mean_mae_255": quality["metrics"]["mean_mae_255"],
                "maximum_motion_relative_error": quality["metrics"]["maximum_motion_relative_error"]})
        gpu = [row["client_wall_seconds"] for row in report["runs"] if row["mode"] == "gpu"]
        hybrid = [row["client_wall_seconds"] for row in report["runs"] if row["mode"] == "hybrid"]
        if (sha256(checkpoint) != MODEL_SHA256 or sha256(text_encoder) != TEXT_SHA256 or
                sha256(tokenizer) != report["tokenizer_sha256"] or sha256(probe) != report["probe_sha256"] or
                sha256(image) != report["input_sha256"] or sha256(manifest) != report["manifest_sha256"] or
                any(sha256(manifest.parent / name) != digest for name, digest in files.items())):
            raise ValueError("source or graph identity changed during measurement")
        report.update(status="complete", gpu_median_seconds=statistics.median(gpu), hybrid_median_seconds=statistics.median(hybrid),
                      matched_speedups=[a / b for a, b in zip(gpu, hybrid)],
                      all_hybrid_faster=all(b < a for a, b in zip(gpu, hybrid)),
                      all_quality_gates_passed=all(row["aligned_rgb_gate_passed"] for row in report["quality"]))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("gpu_median_seconds", "hybrid_median_seconds", "matched_speedups", "all_hybrid_faster", "all_quality_gates_passed", "quality")}, indent=2))


if __name__ == "__main__":
    main()
