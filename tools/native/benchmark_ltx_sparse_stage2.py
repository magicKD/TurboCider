#!/usr/bin/env python3
"""Paired production Stage-2 sparse experiments, with unchanged dense Stage-1."""
import argparse
import ctypes as c
import json
import os
import re
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from benchmark_ltx_resident import load_library, create_engine, consume, bf16_metrics
from benchmark_ltx_env_abba import file_sha256, sequence
from video_quality_gate import compare, passes


def validate_ane_stage2_profile(profile):
    """Admit only MLP-only Stage-2 changes; native preflight checks artifacts."""
    if (not isinstance(profile, dict) or
            profile.get("schema") != "turbocider-ltx-ane-v1" or
            type(profile.get("mlp_stage_mask")) is not int or
            profile["mlp_stage_mask"] != 2 or
            type(profile.get("kv_stage_mask", 0)) is not int or
            profile.get("kv_stage_mask", 0) != 0 or
            any(profile.get(key) for key in
                ("qkv_stage1", "qkv_stage2", "v2a_stage1", "v2a_stage2", "kv"))):
        raise ValueError("ANE paired experiment requires isolated Stage-2 MLP-only profile")


def validate_requests(library, base, candidate):
    """Validate both variants without allocating an engine or running a GPU."""
    library.tc_plan_json.argtypes = [c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
    library.tc_plan_json.restype = c.c_int
    plans = {}
    for variant, options in (("baseline", {}), ("candidate", candidate)):
        value, error = c.c_void_p(), c.c_void_p()
        status = library.tc_plan_json(json.dumps(dict(base, **options)).encode(),
                                      c.byref(value), c.byref(error))
        raw, failure = consume(library, value), consume(library, error)
        if status:
            raise RuntimeError(f"{variant} preflight failed: {failure}")
        plans[variant] = json.loads(raw)
    return plans


def parse_native_profile(text):
    """Keep individual stage records; never sum overlapping AV branch times."""
    profiles = []
    for line in text.splitlines():
        if not line.startswith("ltx_c_profile "):
            continue
        values = dict(re.findall(r"([a-zA-Z0-9_]+)=([0-9.eE+-]+)", line))
        stage = int(values.pop("stage"))
        profiles.append(dict(stage=stage, **{key: float(value) for key, value in values.items()}))
    return profiles


@contextmanager
def capture_native_stderr(path):
    """C stdio bypasses Python redirect_stderr; retain native diagnostics."""
    saved = os.dup(2)
    try:
        with path.open("wb") as log:
            os.dup2(log.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A cinematic red fox running through a snowy forest")
    parser.add_argument("--mode", type=int, choices=(0, 1, 2, 3, 4, 5), default=2)
    parser.add_argument("--keep-blocks", type=int, default=16)
    parser.add_argument("--block-window", action="store_true", help="radius in blocks, not latent frames")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--anchor-stride", type=int, default=16)
    parser.add_argument("--tau", type=float, default=.5)
    parser.add_argument("--dense-edge-steps", type=int, choices=range(17), default=0,
                        help="retain dense first/last N steps; 1 admits only the middle Stage-2 step")
    parser.add_argument("--dense-edge-blocks", type=int, choices=range(25), default=1,
                        help="retain dense first/last N transformer blocks")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--pilot", action="store_true", help="one measured AB pair; not ABBA qualification")
    parser.add_argument("--skip-rgb", action="store_true", help="defer expensive decoded RGB metrics")
    parser.add_argument("--profile", action="store_true", help="record native per-stage operator times and stderr")
    parser.add_argument("--ane-manifest", type=Path,
                        help="test a Stage-2-only ANE MLP profile instead of sparse attention")
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("rounds must be positive")
    root = Path(__file__).resolve().parents[2]
    lib_path = root / "build/native/libturbocider.dylib"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        parser.error("output already has a report; choose a new output directory")
    base = dict(schema_version=1, model="ltx-2.5-distilled", operation="video.generate",
                prompt=args.prompt, width=args.width, height=args.height,
                frames=args.frames, fps=24, steps=11, seed=args.seed,
                execution="gpu", residency="component_staged", audio=True,
                ltx_backend="c_metal", ltx_fast_av=True)
    candidate = dict(allow_approximation=True, ltx_sol_stage2=True,
                     ltx_sol_dense_edge_blocks=args.dense_edge_blocks,
                     ltx_sol_dense_edge_steps=args.dense_edge_steps,
                     ltx_sol_tau=args.tau)
    if args.mode:
        candidate.update(ltx_sparse_mode=args.mode, ltx_sparse_radius=args.radius,
                         ltx_sparse_anchor_stride=args.anchor_stride,
                         ltx_sparse_tokens_per_frame=0 if args.block_window else (args.width//32)*(args.height//32))
        if args.mode in (4, 5):
            candidate["ltx_sparse_keep_blocks"] = args.keep_blocks
    ane_profile = None
    if args.ane_manifest:
        ane_profile = json.loads(args.ane_manifest.read_text())
        try:
            validate_ane_stage2_profile(ane_profile)
        except ValueError as error:
            parser.error(str(error))
        candidate = dict(execution="gpu_ane", allow_approximation=True,
                         ane_manifest=str(args.ane_manifest.resolve()))
    report = dict(schema="ltx-sparse-stage2-paired-v1", workload=base,
                  candidate=candidate, library_sha256=file_sha256(lib_path),
                  shader_sha256=file_sha256(root / "build/native/ltx_shaders.metal"),
                  warmups=[], runs=[], complete=False,
                  native_profile_enabled=args.profile,
                  method="retained engine, one warmup each, AB pilot" if args.pilot else
                         "retained engine, one warmup each, alternating ABBA/BAAB")
    if ane_profile is not None:
        report.update(schema="ltx-ane-stage2-paired-v1", ane_profile=ane_profile,
                      ane_profile_sha256=file_sha256(args.ane_manifest))
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    library = load_library(lib_path)
    # Reject stale binaries/unsupported candidates before expensive GPU warmups.
    try:
        report["validated_plans"] = validate_requests(library, base, candidate)
    except RuntimeError as error:
        report["preflight_error"] = str(error)
        save()
        raise
    save()
    engine = create_engine(library, args.model.resolve())
    original_profile = os.environ.get("TURBOCIDER_LTX_C_PROFILE")
    # Explicitly disable ambient profiling for unprofiled measurements.
    os.environ["TURBOCIDER_LTX_C_PROFILE"] = "1" if args.profile else "0"
    try:
        for phase, variants in (("warmups", ["baseline", "candidate"]),
                                ("runs", ["baseline", "candidate"] if args.pilot else sequence(args.rounds))):
            for index, variant in enumerate(variants):
                stem = output / f"{phase}-{index}-{variant}"
                dump = stem.with_name(stem.name + "-tensors")
                request = dict(base, output=str(stem.with_suffix(".mp4")), dump_tensors=str(dump))
                if variant == "candidate":
                    request.update(candidate)
                value, error = c.c_void_p(), c.c_void_p()
                started = time.perf_counter()
                log = stem.with_suffix(".stderr.log")
                with capture_native_stderr(log) if args.profile else nullcontext():
                    status = library.tc_engine_generate(engine, json.dumps(request).encode(), None, None,
                                                        c.byref(value), c.byref(error))
                raw, failure = consume(library,value), consume(library,error)
                if status:
                    raise RuntimeError(failure)
                result = json.loads(raw)
                row = dict(variant=variant, request=request, result=result,
                           client_wall_seconds=time.perf_counter()-started)
                if args.profile:
                    row["native_stderr"] = str(log)
                    row["native_profiles"] = parse_native_profile(log.read_text(errors="replace"))
                    if {p["stage"] for p in row["native_profiles"]} != {1, 2}:
                        raise RuntimeError(f"missing native profile stages; inspect {log}")
                report[phase].append(row)
                save()
                print(json.dumps(dict(phase=phase, variant=variant,
                                      timings=result["timings_seconds"])), flush=True)
    finally:
        library.tc_engine_free(engine)
        if original_profile is None:
            os.environ.pop("TURBOCIDER_LTX_C_PROFILE", None)
        else:
            os.environ["TURBOCIDER_LTX_C_PROFILE"] = original_profile
    baseline = [r for r in report["runs"] if r["variant"] == "baseline"]
    candidates = [r for r in report["runs"] if r["variant"] == "candidate"]
    report["speedup"] = {
        field: statistics.median(r["result"]["timings_seconds"][field] for r in baseline) /
               statistics.median(r["result"]["timings_seconds"][field] for r in candidates)
        for field in ("stage1", "stage2", "request_wall")
    }
    report["quality"] = []
    reference = baseline[0]["request"]
    for row in candidates:
        current = row["request"]
        metrics = {name: bf16_metrics(Path(reference["dump_tensors"])/name,
                                      Path(current["dump_tensors"])/name)
                   for name in ("stage1_video.bf16", "stage2_input_video.bf16", "stage2_video.bf16")}
        if not metrics["stage1_video.bf16"]["byte_exact"] or not metrics["stage2_input_video.bf16"]["byte_exact"]:
            raise RuntimeError("Stage-2 isolation failed: dense Stage-1 or Stage-2 input differs")
        if not args.skip_rgb:
            metrics["rgb"] = compare(Path(reference["output"]), Path(current["output"]))
            metrics["rgb_gate_passed"] = passes(metrics["rgb"])
        report["quality"].append(metrics)
        save()
    report["complete"] = True
    save()
    print(json.dumps(report["speedup"]), flush=True)


if __name__ == "__main__":
    main()
