#!/usr/bin/env python3
"""Read-only vpipe Qwen2.1 DiT comparison. NOT complete-request performance."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

import collect
import vpipe_view
from collect import Observer, ROOT, model_snapshot, sha256_file, write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES = tuple("tools/profiling/qwen_base/" + name for name in
                ("vpipe_compare.py", "vpipe_probe.cpp", "vpipe_probe.cmake", "vpipe_view.py", "observer.mm", "collect.py"))
ARMS = dict(gpu=0, ffn=1, **{"ffn-qkv": 2})


def reference_identity(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    if git("status", "--porcelain").strip():
        raise ValueError("reference checkout must remain clean")
    paths = [p.decode() for p in git("ls-files", "-z").split(b"\0") if p]
    return dict(commit=git("rev-parse", "HEAD").decode().strip(), submodules=git("submodule", "status", "--recursive").decode(),
                files={name: sha256_file(root / name) for name in paths if (root / name).is_file()})


def environment(mode):
    env = {k: v for k, v in benchmark_environment().items() if not k.startswith("VPIPE_")}
    # This is diagnostic forwarding, not a pipeline launch or registry write.
    if mode == "timeline":
        env["VPIPE_QWEN_IMAGE21_ANE_PROFILE"] = "1"
    return env


def validate_report(report, case, arm, reps):
    expected = (5184, 4096) if case == "edit" else (1088, 0)
    if (report.get("schema") != "tc-vpipe-qwen-dit-v1" or report.get("status") != "complete" or
            report.get("complete_request") is not False or report.get("synthetic_inputs") is not True or
            (report.get("joint_rows"), report.get("reference_rows")) != expected or
            report.get("decode_rows") != 1024 or report.get("arm") != ARMS[arm] or
            report.get("edit") != int(case == "edit")):
        raise ValueError("complete matched base DiT component geometry required")
    samples = report.get("samples", [])
    if len(samples) != 2 * (reps + 1):
        raise ValueError("separate warmup plus all measured prefill/decode pairs required")
    for index, row in enumerate(samples):
        if (row.get("index") != index or row.get("rep") != index // 2 or
                row.get("phase") != ("decode" if index % 2 else "prefill") or
                row.get("warmup") is not (index < 2) or
                type(row.get("seconds")) not in (int, float) or
                not math.isfinite(row["seconds"]) or row["seconds"] <= 0 or
                row.get("ffn_failed") is not False or row.get("qkv_failed") is not False):
            raise ValueError("complete finite samples without silent runtime failure required")
        for name in ("gpu_flops", "ane_flops", "ane_predict_records"):
            if type(row.get(name)) is not int or row[name] < 0:
                raise ValueError("actual arithmetic/predict counters required")
        if not row["gpu_flops"] or bool(row["ane_flops"]) != bool(row["ane_predict_records"]):
            raise ValueError("inconsistent actual arithmetic credit")
        if arm == "gpu" or case == "generate":
            if row["ffn_armed"] or row["qkv_armed"] or row["ane_flops"]:
                raise ValueError("default vpipe512 generation must not be mislabeled active ANE")
        elif not row["ffn_armed"] or row["qkv_armed"] != (arm == "ffn-qkv"):
            raise ValueError("requested vpipe long-prefill module did not arm")
        if index % 2 and row["ane_flops"]:
            raise ValueError("2048-row reference module cannot offload1024-row decode")
    if case == "edit" and arm != "gpu" and not any(row["ane_flops"] for row in samples if row["phase"] == "prefill"):
        raise ValueError("ANE-enabled long-prefill comparison requires actual ANE work")
    return {phase: dict(seconds=[r["seconds"] for r in samples if r["phase"] == phase and not r["warmup"]],
                        median_seconds=statistics.median(r["seconds"] for r in samples if r["phase"] == phase and not r["warmup"]),
                        ane_predict_records=[r["ane_predict_records"] for r in samples if r["phase"] == phase])
            for phase in ("prefill", "decode")}


def validate_arrays(root, report):
    arrays = []
    for row in report["samples"]:
        path = root / f"sample-{row['index']}.f32"
        value = np.fromfile(path, dtype="<f4")
        if value.size != 1024 * 64 or not np.isfinite(value).all():
            raise ValueError("complete finite target output required")
        arrays.append(value)
    # Fixed synthetic inputs + timestep + fresh prefix fill, no stale replay.
    # Auto row selection can change mixed fp16/BF16 rounding, so only compare
    # repeatability exactly when the reference declined ANE entirely.
    if not any(r["ane_flops"] for r in report["samples"]):
        if any(not np.array_equal(arrays[i % 2], value) for i, value in enumerate(arrays)):
            raise ValueError("unchanged GPU synthetic forward did not restore output")
    return {f"sample-{i}.f32": sha256_file(root / f"sample-{i}.f32") for i in range(len(arrays))}


def child(args):
    observer = Observer(args.observer) if args.mode == "timeline" else None
    lib = C.CDLL(str(args.library))
    scope_type = C.CFUNCTYPE(None, C.c_int)
    scope = scope_type(lambda i: observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    function = lib.tc_vpipe_qwen_probe
    function.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, scope_type]
    function.restype = C.c_int
    try:
        code = function(str(args.checkpoint).encode(), int(args.case == "edit"), ARMS[args.arm], args.reps,
                        str(args.output).encode(), scope)
        if code:
            raise RuntimeError(f"vpipe component probe returned{code}; inspect stderr")
    finally:
        if observer:
            write_json(args.output / "observer.json", observer.take("tc_qwen_profile_finish"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "references/vpipe")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1/diffusion_models/qwen_image_2.1_bf16.safetensors")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1", help="original weights for bounded source identity")
    parser.add_argument("--observer", type=Path, default=ROOT / "build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--case", choices=("generate", "edit"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--mode", choices=("stock", "timeline"), default="stock")
    parser.add_argument("--reps", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    for key in ("reference_root", "library", "checkpoint", "observer", "model"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.child:
        child(args)
        return
    identity = reference_identity(args.reference_root)
    view_identity = vpipe_view.validate(args.model, args.checkpoint) if args.checkpoint.is_dir() else None
    hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    build = args.library.parent
    # The fresh reference build is kept outside the checkout. Bind its full
    # compile/link recipe; an unrelated existing libvpipe is never substituted.
    recipes = [build / "CMakeCache.txt", build / "compile_commands.json",
               build / "CMakeFiles/qwen_vpipe_reference.dir/link.txt"]
    args.output.mkdir(parents=True, exist_ok=False)
    trial = args.output / "trial"
    trial.mkdir()
    summary = dict(schema="tc-vpipe-qwen-component-screen-v1", status="running", qualification_passed=False,
                   complete_request=False, synthetic_inputs=True, case=args.case, arm=args.arm, mode=args.mode, reps=args.reps,
                   reference=identity, sources=hashes, library_sha256=sha256_file(args.library),
                   build_recipes={str(path.relative_to(build)): dict(sha256=sha256_file(path), text=path.read_text()) for path in recipes},
                   checkpoint_snapshot=model_snapshot(args.model),
                   checkpoint_view=json.loads((args.checkpoint / "view.json").read_text()) if args.checkpoint.is_dir() else None,
                   view_identity=view_identity,
                   scope="DiT only: synthetic64 text rows,512 target, optional1024 reference; no encoder/sampler/VAE/PNG; no production speed qualification")
    write_json(args.output / "source.json", {name: (ROOT / name).read_text() for name in SOURCES})
    write_json(args.output / "summary.json", summary)
    load = LoadObservation(args.output / "load.jsonl")
    try:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--reference-root", str(args.reference_root),
                   "--library", str(args.library), "--checkpoint", str(args.checkpoint), "--model", str(args.model), "--observer", str(args.observer),
                   "--case", args.case, "--arm", args.arm, "--mode", args.mode, "--reps", str(args.reps), "--output", str(trial)]
        with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=environment(args.mode),
                                            stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500, observer=load)
        report = json.loads((trial / "report.json").read_text())
        summary["analysis"] = validate_report(report, args.case, args.arm, args.reps)
        summary["arrays"] = validate_arrays(trial, report)
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    try:
        summary["load"] = load.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    if (reference_identity(args.reference_root) != identity or model_snapshot(args.model) != summary["checkpoint_snapshot"] or
            any(sha256_file(ROOT / name) != digest for name, digest in hashes.items()) or
            (view_identity is not None and vpipe_view.validate(args.model, args.checkpoint) != view_identity)):
        summary.update(status="failed", error="bound local/reference source changed")
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("reference", "build_recipes", "checkpoint_snapshot")}, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
