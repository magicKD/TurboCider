#!/usr/bin/env python3
"""Probe cold and cached LTX 0.9.8 GPU/Core ML requests in A/B/B/A order.

Each process issues two requests on one pipeline with one prompt and consecutive
seeds. This is an isolated measurement, not a production route or evidence of
physical ANE placement. All videos are encoded by the same native encoder.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from ltx098_comfy_reference import MODEL_NAME, MODEL_SHA256, TEXT_NAME, TEXT_SHA256, sha256


def graph_receipt(manifest):
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
    return compiled


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def validate_detail(detail, mode, request_index, args):
    expected_hybrid = 28 * args.step_mask.bit_count() if mode == "hybrid" else 0
    expected = (mode, args.seed + request_index, request_index, 2, 8,
                224 if mode == "hybrid" else 0, expected_hybrid,
                expected_hybrid * args.chunks, bool(request_index), bool(request_index))
    actual = tuple(detail.get(key) for key in (
        "mode", "seed", "request_index", "requests_in_process", "actual_steps", "ffn_calls",
        "hybrid_ffn_calls", "coreml_calls", "text_cache_hit", "cross_attention_cache_hit"))
    if actual != expected or detail.get("hybrid_step_mask") != args.step_mask:
        raise ValueError(f"{mode} request {request_index} call/cache contract differs: {actual}")
    if mode == "hybrid" and (detail["per_block_calls"] != [8] * 28 or
            detail["per_step_ffn_calls"] != [28] * 8 or
            detail["per_step_hybrid_ffn_calls"] != [28 if args.step_mask & (1 << i) else 0 for i in range(8)]):
        raise ValueError(f"{mode} request {request_index} block/step calls differ")
    return expected_hybrid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "model", "input", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--step-mask", type=int, default=192)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="finish quality checks on a failed run with all eight request reports, writing report-final.json")
    args = parser.parse_args()
    probe, model, image, manifest = (p.resolve(strict=True) for p in (args.probe, args.model, args.input, args.manifest))
    output = args.output.resolve()
    if args.finalize_existing != output.is_dir() or (not args.finalize_existing and output.exists()):
        parser.error("output must be new, or an existing directory with --finalize-existing")
    if not args.prompt or not 0 <= args.seed < 0xffffffffffffffff or not 1 <= args.chunks <= 15 or not 1 <= args.step_mask <= 255:
        parser.error("prompt, uint64 seed pair, 1–15 chunks and 1–255 step mask required")
    if any(importlib.util.find_spec(name) is None for name in ("torch", "safetensors")):
        parser.error("this validation needs an existing Python environment with torch and safetensors")
    checkpoint, text_encoder = model / MODEL_NAME, model / "text_encoders" / TEXT_NAME
    tokenizer = model / "tokenizer/tokenizer.json"
    if sha256(checkpoint) != MODEL_SHA256 or sha256(text_encoder) != TEXT_SHA256:
        raise ValueError("checkpoint or text encoder SHA-256 differs from pinned source")
    compiled = graph_receipt(manifest)
    identities = {"checkpoint_sha256": MODEL_SHA256, "text_encoder_sha256": TEXT_SHA256,
                  "tokenizer_sha256": sha256(tokenizer), "manifest_sha256": sha256(manifest),
                  "probe_sha256": sha256(probe), "input_sha256": sha256(image)}
    order = (("a0", "gpu"), ("b0", "hybrid"), ("b1", "hybrid"), ("a1", "gpu"))
    if args.finalize_existing:
        report = json.loads((output / "report.json").read_text())
        if (report.get("schema") != "ltx098-ane-hybrid-resident-abba-v1" or report.get("status") != "failed" or
                any(report.get(key) != value for key, value in identities.items()) or
                (report.get("prompt"), report.get("chunks"), report.get("step_mask")) != (args.prompt, args.chunks, args.step_mask) or
                report.get("quality") != [] or len(report.get("runs", [])) != 8 or
                (output / "report-final.json").exists() or any(output.glob("quality-*-*.json"))):
            raise ValueError("existing report is not a matching, unfinished eight-request run")
        for row, (label, mode, request_index) in zip(report["runs"],
                ((label, mode, request_index) for label, mode in order for request_index in range(2))):
            detail = json.loads((output / label / f"request-{request_index}/report.json").read_text())
            validate_detail(detail, mode, request_index, args)
            if (row.get("label"), row.get("mode"), row.get("request_index"), row.get("seed"), row.get("wall_seconds")) != (
                    label, mode, request_index, args.seed + request_index, detail["wall_seconds"]):
                raise ValueError("existing run and request receipt disagree")
        report["resumed_from_error"] = report.pop("error", None)
        report["status"] = "running"
        report_path = output / "report-final.json"
    else:
        output.mkdir(parents=True)
        report_path = output / "report.json"
        report = {"schema": "ltx098-ane-hybrid-resident-abba-v1", "status": "running", **identities,
                  "prompt": args.prompt, "chunks": args.chunks, "step_mask": args.step_mask,
                  "scope": "Four fresh processes A GPU / B CPU_AND_NE+GPU / B / A; each process issues "
                           "two full requests, same prompt, consecutive seeds, 512x512x121, 8 steps, same encoder. "
                           "First in-process time includes initialization; second reuses pipeline and caches. "
                           "Process startup is excluded. Plan is not runtime placement proof.",
                  "physical_ane_placement": "unknown", "runs": [], "quality": []}
    try:
        for label, mode in (() if args.finalize_existing else order):
            command = [str(probe), str(model), str(image), str(compiled), str(output / label), args.prompt,
                       str(args.seed), mode, str(args.chunks), str(args.step_mask), "2"]
            started = time.perf_counter()
            run = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            client_wall = time.perf_counter() - started
            (output / f"{label}.stdout.txt").write_text(run.stdout)
            (output / f"{label}.stderr.txt").write_text(run.stderr)
            if run.returncode:
                raise RuntimeError(f"{label} exited {run.returncode}: {run.stderr[-4000:]}")
            for request_index in range(2):
                detail = json.loads((output / label / f"request-{request_index}/report.json").read_text())
                expected_hybrid = validate_detail(detail, mode, request_index, args)
                row = {"label": label, "mode": mode, "request_index": request_index,
                       "seed": detail["seed"], "wall_seconds": detail["wall_seconds"],
                       "denoise_seconds": detail["denoise_seconds"], "text_seconds": detail["text_seconds"],
                       "cross_attention_prepare_seconds": detail["cross_attention_prepare_seconds"],
                       "text_cache_hit": detail["text_cache_hit"], "cross_attention_cache_hit": detail["cross_attention_cache_hit"],
                       "graph_load_and_selftest_seconds": detail["graph_load_and_selftest_seconds"],
                       "hybrid_ffn_calls": expected_hybrid, "coreml_calls": detail["coreml_calls"],
                       "process_client_wall_seconds": client_wall}
                report["runs"].append(row)
                print(json.dumps(row), flush=True)
            write_report(report_path, report)

        import torch
        from safetensors.torch import load_file
        for pair_index in (0, 1):
            for request_index in (0, 1):
                ref_dir = output / f"a{pair_index}/request-{request_index}"
                candidate_dir = output / f"b{pair_index}/request-{request_index}"
                reference = load_file(str(ref_dir / "final-latent.safetensors"))["value"].float()
                candidate = load_file(str(candidate_dir / "final-latent.safetensors"))["value"].float()
                if reference.shape != candidate.shape or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
                    raise ValueError("invalid reference or hybrid final latent")
                quality_path = output / f"quality-{pair_index}-{request_index}.json"
                compare = subprocess.run([sys.executable, str(Path(__file__).with_name("ltx098_video_comparison.py")),
                    "--reference", str(ref_dir / "0.mp4"), "--candidate", str(candidate_dir / "0.mp4"),
                    "--output", str(quality_path)], capture_output=True, text=True, timeout=600)
                (output / f"quality-{pair_index}-{request_index}.stdout.txt").write_text(compare.stdout)
                (output / f"quality-{pair_index}-{request_index}.stderr.txt").write_text(compare.stderr)
                if compare.returncode:
                    raise RuntimeError(compare.stderr)
                quality = json.loads(quality_path.read_text())
                row = {"pair_index": pair_index, "request_index": request_index, "seed": args.seed + request_index,
                       "latent_relative_l2": ((reference - candidate).norm() / reference.norm().clamp_min(1e-30)).item(),
                       "contract_passed": quality["contract_passed"], "aligned_rgb_gate_passed": quality["aligned_rgb_gate_passed"],
                       "mean_correlation": quality["metrics"]["mean_correlation"], "mean_mae_255": quality["metrics"]["mean_mae_255"],
                       "maximum_motion_relative_error": quality["metrics"]["maximum_motion_relative_error"]}
                report["quality"].append(row)
                write_report(report_path, report)
                print(json.dumps(row), flush=True)
        if (sha256(checkpoint) != identities["checkpoint_sha256"] or sha256(text_encoder) != identities["text_encoder_sha256"] or
                sha256(tokenizer) != identities["tokenizer_sha256"] or sha256(probe) != identities["probe_sha256"] or
                sha256(image) != identities["input_sha256"] or sha256(manifest) != identities["manifest_sha256"] or
                graph_receipt(manifest) != compiled):
            raise ValueError("source or graph identity changed during measurement")
        results = {}
        for request_index, phase in enumerate(("cold", "warm")):
            selected = [r for r in report["runs"] if r["request_index"] == request_index]
            gpu = [r["wall_seconds"] for r in selected if r["mode"] == "gpu"]
            hybrid = [r["wall_seconds"] for r in selected if r["mode"] == "hybrid"]
            results[phase] = {"gpu_median_seconds": statistics.median(gpu), "hybrid_median_seconds": statistics.median(hybrid),
                              "matched_speedups": [a / b for a, b in zip(gpu, hybrid)],
                              "all_hybrid_faster": all(b < a for a, b in zip(gpu, hybrid))}
        report.update(status="complete", phases=results,
                      all_quality_gates_passed=all(row["contract_passed"] and row["aligned_rgb_gate_passed"] for row in report["quality"]))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_report(report_path, report)
    print(json.dumps({"phases": report["phases"], "all_quality_gates_passed": report["all_quality_gates_passed"]}, indent=2))


if __name__ == "__main__":
    main()
