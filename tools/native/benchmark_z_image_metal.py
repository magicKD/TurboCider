"""Fresh-process ABBA, persistent warm sessions, pure GPU diffusion timing.

The first request of each process is excluded. Dump validation is a separate
request and is never included in the speedup. All native metrics are retained.
"""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time


# These native switches are tested with getenv(), not parsed as booleans.
# Reject misleading values rather than silently benchmarking an enabled path.
PRESENCE_FLAGS = frozenset({
    "TURBOCIDER_Z_MPP_SWIGLU", "TURBOCIDER_Z_MPP_SWIGLU_DUAL",
    "TURBOCIDER_Z_FUSED_GATE_NORM", "TURBOCIDER_Z_DISABLE_FUSED_QKV",
    "TURBOCIDER_Z_DISABLE_FUSED_MOD", "TURBOCIDER_Z_DISABLE_FUSED_SDPA",
    "TURBOCIDER_Z_DISABLE_VECTOR_NORM", "TURBOCIDER_Z_INLINE_GATE_TANH",
    "TURBOCIDER_Z_EAGER_ROPE",
    "TURBOCIDER_Z_DISABLE_MPP_SWIGLU",
    "TURBOCIDER_Z_DISABLE_VIRTUAL_NORM",
    "TURBOCIDER_Z_CACHE_CONTEXT",
    "TURBOCIDER_Z_MPP_QKV_PREPARE",
    "TURBOCIDER_Z_SWIGLU_LEGACY_32X256",
    "TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS",
    "TURBOCIDER_Z_DISABLE_MPP_QKV_PREPARE",
    "TURBOCIDER_Z_DISABLE_GATE_NORM",
    "TURBOCIDER_Z_DISABLE_CACHE_CONTEXT",
})


def validate_presence_flags(environment):
    for key in sorted(PRESENCE_FLAGS & environment.keys()):
        if environment[key].strip().lower() in ("", "0", "false", "off", "no"):
            raise ValueError(f"{key} is presence-based: unset it to disable; "
                             "a false-looking value still enables it")


def loaded_runtime_libraries():
    """Record actual dyld images, not merely the requested search path."""
    if sys.platform != "darwin":
        return {}
    dyld = C.CDLL(None)
    dyld._dyld_image_count.argtypes = []
    dyld._dyld_image_count.restype = C.c_uint32
    dyld._dyld_get_image_name.argtypes = [C.c_uint32]
    dyld._dyld_get_image_name.restype = C.c_char_p
    libraries = {}
    for i in range(dyld._dyld_image_count()):
        name = dyld._dyld_get_image_name(i)
        if not name:
            continue
        path = Path(os.fsdecode(name))
        if path.name in ("libmlx.dylib", "libjaccl.dylib", "libturbocider.dylib"):
            libraries[path.name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return libraries


def system_state_metadata():
    """Cheap provenance for detecting cross-boot or obviously degraded runs."""
    boot_time = None
    try:
        result = subprocess.run(["sysctl", "-n", "kern.boottime"], check=True,
                                capture_output=True, text=True)
        boot_time = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"boot_time": boot_time, "monotonic_uptime_seconds": time.monotonic(),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}


def assess_system_state(reports, conditions):
    """Fail closed for the qualified M4 Max anchor workloads.

    The pre-reboot degraded state was roughly 2x slow in DiT and >20x slow in
    VAE while still producing stable ABBA ratios. A matched schedule alone
    therefore cannot qualify performance. These deliberately loose ceilings
    distinguish that state from normal run noise without becoming targets.
    """
    if not reports or reports[0].get("hardware", {}).get("gpu") != "Apple M4 Max":
        return {"status": "not_applicable", "reason": "unqualified hardware"}
    thresholds = {(512, 8): (9.0, 1.0), (1024, 9): (45.0, 2.0)}
    limit = thresholds.get((conditions["size"], conditions["steps"]))
    if not limit:
        return {"status": "not_applicable", "reason": "no anchor for workload"}
    variants = {report["variant"] for report in reports}
    reference = "baseline" if "baseline" in variants else "fused"
    rows = [row for report in reports if report["variant"] == reference
            for row in report["runs"] if not row["warmup"] and not row["parity"]]
    if not rows:
        return {"status": "invalid", "reason": "missing warm anchor samples"}
    denoise = statistics.median(row["metrics"]["timings_seconds"]["denoise"] for row in rows)
    vae = statistics.median(row["metrics"]["timings_seconds"]["vae_decode"] for row in rows)
    healthy = denoise <= limit[0] and vae <= limit[1]
    return {"status": "healthy" if healthy else "invalid_for_performance",
            "reference_variant": reference, "median_denoise_seconds": denoise,
            "median_vae_seconds": vae,
            "maximum_denoise_seconds": limit[0], "maximum_vae_seconds": limit[1],
            "reason": "qualified anchor within loose health ceiling" if healthy else
                      "system/runtime state is outside the qualified health envelope"}


def summarize_reports(reports):
    """Keep wall and diffusion medians separate; exclude cold/dump requests."""
    grouped = {}
    for report in reports:
        bucket = grouped.setdefault(report["variant"], [])
        bucket.extend(row for row in report["runs"]
                      if not row["warmup"] and not row["parity"])
    result = {}
    for variant, rows in grouped.items():
        if not rows:
            continue
        result[variant] = {
            "warm_samples": len(rows),
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
            "median_timings_seconds": {
                key: statistics.median(row["metrics"]["timings_seconds"][key] for row in rows)
                for key in rows[0]["metrics"]["timings_seconds"]},
        }
    if "baseline" in result and "fused" in result:
        result["wall_speedup"] = (result["baseline"]["median_wall_seconds"] /
                                  result["fused"]["median_wall_seconds"])
    return result


def worker(a):
    # Same advisory lock as the C++ probes. Hold across cold/warm/quality
    # requests so a probe cannot contaminate an in-progress process.
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return locked_worker(a)


def locked_worker(a):
    fingerprint = hashlib.sha256(Path(a.library).read_bytes()).hexdigest()
    lib = C.CDLL(str(Path(a.library).resolve()))
    ptr = C.c_void_p
    lib.tc_string_free.argtypes = [ptr]
    lib.tc_system_json.argtypes = []
    lib.tc_system_json.restype = ptr
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p,
                                          C.POINTER(ptr), C.POINTER(ptr)]
    lib.tc_engine_generate.argtypes = [ptr, C.c_char_p, ptr, ptr,
                                       C.POINTER(ptr), C.POINTER(ptr)]
    lib.tc_engine_free.argtypes = [ptr]

    def take(v):
        if not v.value:
            return None
        text = C.string_at(v).decode()
        lib.tc_string_free(v)
        return text

    hardware = json.loads(take(ptr(lib.tc_system_json())))
    engine, err = ptr(), ptr()
    status = lib.tc_engine_create_model(b"z-image-turbo", a.model.encode(),
                                         C.byref(engine), C.byref(err))
    if status:
        raise RuntimeError(take(err))
    report = {"variant": a.worker, "library_sha256": fingerprint, "environment": {
        k: v for k, v in os.environ.items()
        if k.startswith(("TURBOCIDER_Z_", "MLX_", "DYLD_", "MTL_", "METAL_"))},
        "runtime_libraries": loaded_runtime_libraries(), "hardware": hardware,
        "system_state": system_state_metadata(),
        "platform": {"macos": platform.mac_ver()[0], "kernel": platform.release(),
                     "machine": platform.machine()}, "runs": []}
    try:
        for i in range(a.runs + 1 + int(a.parity)):
            quality = i == a.runs + 1
            request = dict(model="z-image-turbo", operation="image.generate",
                           prompt=a.prompt, output=str(a.output / f"image-{i}.png"),
                           width=a.size, height=a.size, steps=a.steps, seed=a.seed,
                           execution="gpu", residency="resident", audio=False,
                           dynamic_text=True)
            if quality:
                request["dump_tensors"] = str(a.output / "parity")
            result, err = ptr(), ptr()
            start = time.monotonic()
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(),
                                            None, None, C.byref(result), C.byref(err))
            text, error = take(result), take(err)
            row = {"warmup": i == 0, "parity": quality, "request": request,
                   "wall_seconds": time.monotonic() - start, "error": error,
                   "metrics": json.loads(text) if text else None}
            report["runs"].append(row)
            (a.output / "report.json").write_text(json.dumps(report, indent=2))
            print(a.worker, i, round(row["wall_seconds"], 3), error or "ok", flush=True)
            if status:
                raise RuntimeError(error)
    finally:
        lib.tc_engine_free(engine)
    if hashlib.sha256(Path(a.library).read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("library changed during measurement; discard this run")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--library", default="build/native/libturbocider.dylib")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--prompt", default="A cinematic red fox walking through fresh snow, soft morning light.")
    p.add_argument("--schedule", default="baseline,fused,fused,baseline")
    p.add_argument("--control-default", action="store_true",
                   help="use production defaults for the baseline arm, not unfused kernels")
    p.add_argument("--control-legacy-swiglu", action="store_true",
                   help="use the pre-reboot 32x256 SwiGLU tile in the baseline arm")
    p.add_argument("--parity", action="store_true")
    p.add_argument("--mpp", action="store_true", help="test experimental fused SwiGLU GEMM")
    p.add_argument("--no-mpp", action="store_true", help="disable the default fused SwiGLU GEMM")
    p.add_argument("--mpp-dual", action="store_true", help="test dual gate/up MPP SwiGLU")
    p.add_argument("--mpp-projections", action="store_true", help="test experimental MPP projections")
    p.add_argument("--mpp-attn-out", action="store_true", help="test only MPP attention output projection")
    p.add_argument("--mpp-qkv-prepare", action="store_true",
                   help="fuse MPP QKV projection with norm, RoPE and SDPA layout")
    p.add_argument("--gate-norm", action="store_true", help="fuse gate residual and FFN input norm")
    p.add_argument("--cache-context", action="store_true", help="reuse exact context refinement within a request")
    p.add_argument("--gate-norm-virtual-threads", type=int, choices=(128,256,512),
                   help="small-shape fused gate/FFN norm with exact virtual reductions")
    p.add_argument("--norm-vector", action="store_true", help="test exact 960-thread four-wide norm accesses")
    p.add_argument("--scalar-norm", action="store_true", help="disable default vector norm in the fused variant")
    p.add_argument("--norm-threads", type=int, choices=(128, 256, 512, 960),
                   help="experimental reduction geometry; may change rounding")
    p.add_argument("--virtual-norm-threads", type=int, choices=(128, 256, 512),
                   help="experimental physical threads preserving the 960-thread reduction order")
    p.add_argument("--worker", choices=("baseline", "fused"))
    a = p.parse_args()
    try:
        validate_presence_flags(os.environ)
    except ValueError as error:
        p.error(str(error))
    if a.runs < 1:
        p.error("at least one warm run required")
    if a.mpp_projections and a.mpp_attn_out:
        p.error("choose all projections or attention output only, not both")
    if a.mpp and a.no_mpp:
        p.error("choose MPP enable or disable, not both")
    if a.gate_norm and a.gate_norm_virtual_threads:
        p.error("choose scalar or virtual gate norm, not both")
    if a.norm_vector and a.norm_threads not in (None, 960):
        p.error("vector norm requires 960 threads")
    if a.norm_vector and a.scalar_norm:
        p.error("choose vector norm or scalar norm, not both")
    if a.virtual_norm_threads and (a.scalar_norm or a.norm_threads not in (None, 960)):
        p.error("virtual norm requires the vector path and default reduction order")
    if a.worker and (a.mpp or a.mpp_dual or a.mpp_projections or a.mpp_attn_out
                     or a.gate_norm or a.norm_vector or a.scalar_norm or a.no_mpp
                     or a.norm_threads is not None or a.virtual_norm_threads is not None
                     or a.control_default or a.control_legacy_swiglu
                     or a.gate_norm_virtual_threads or a.cache_context
                     or a.mpp_qkv_prepare):
        p.error("worker mode requires kernel environment variables, not experiment flags")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker(a)
        return
    if any(k.startswith("TURBOCIDER_Z_") for k in os.environ):
        p.error("unset TURBOCIDER_Z_* overrides before a matched experiment")
    samples = {"baseline": [], "fused": []}
    reports = []
    report_data = []
    fingerprint = hashlib.sha256(Path(a.library).read_bytes()).hexdigest()
    for i, variant in enumerate(a.schedule.split(",")):
        if variant not in samples:
            p.error("schedule must contain baseline/fused")
        dest = a.output / f"{i}-{variant}"
        env = dict(os.environ)
        if variant == "baseline" and a.control_legacy_swiglu:
            env["TURBOCIDER_Z_SWIGLU_LEGACY_32X256"] = "1"
        if variant == "baseline" and not a.control_default:
            env["TURBOCIDER_Z_DISABLE_FUSED_QKV"] = "1"
            env["TURBOCIDER_Z_DISABLE_FUSED_MOD"] = "1"
            env["TURBOCIDER_Z_DISABLE_MPP_SWIGLU"] = "1"
            env["TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS"] = "1"
            env["TURBOCIDER_Z_DISABLE_MPP_QKV_PREPARE"] = "1"
            env["TURBOCIDER_Z_DISABLE_GATE_NORM"] = "1"
            env["TURBOCIDER_Z_DISABLE_CACHE_CONTEXT"] = "1"
        elif variant == "fused" and a.mpp:
            env["TURBOCIDER_Z_MPP_SWIGLU"] = "1"
        if variant == "fused" and a.no_mpp:
            env["TURBOCIDER_Z_DISABLE_MPP_SWIGLU"] = "1"
        if variant == "fused" and a.mpp_dual:
            env["TURBOCIDER_Z_MPP_SWIGLU_DUAL"] = "1"
        if variant == "fused" and a.mpp_projections:
            env["TURBOCIDER_Z_MPP_PROJECTIONS"] = "1"
        if variant == "fused" and a.mpp_attn_out:
            env["TURBOCIDER_Z_MPP_PROJECTIONS"] = "attention_out"
        if variant == "fused" and a.mpp_qkv_prepare:
            env["TURBOCIDER_Z_MPP_QKV_PREPARE"] = "1"
        if variant == "fused" and a.gate_norm:
            env["TURBOCIDER_Z_FUSED_GATE_NORM"] = "1"
        if variant == "fused" and a.cache_context:
            env["TURBOCIDER_Z_CACHE_CONTEXT"] = "1"
        if variant == "fused" and a.gate_norm_virtual_threads:
            env["TURBOCIDER_Z_GATE_NORM_VIRTUAL_THREADS"] = str(a.gate_norm_virtual_threads)
        if variant == "fused" and a.norm_vector:
            env["TURBOCIDER_Z_NORM_VECTOR"] = "1"
        if variant == "fused" and a.scalar_norm:
            env["TURBOCIDER_Z_DISABLE_VECTOR_NORM"] = "1"
        if variant == "fused" and a.norm_threads is not None:
            env["TURBOCIDER_Z_NORM_THREADS"] = str(a.norm_threads)
        if variant == "fused" and a.virtual_norm_threads is not None:
            env["TURBOCIDER_Z_VIRTUAL_NORM_THREADS"] = str(a.virtual_norm_threads)
        cmd = [sys.executable, __file__, "--worker", variant, "--model", a.model,
               "--library", a.library, "--output", str(dest), "--size", str(a.size),
               "--steps", str(a.steps), "--seed", str(a.seed), "--runs", str(a.runs),
               "--prompt", a.prompt]
        if a.parity:
            cmd.append("--parity")
        subprocess.run(cmd, env=env, check=True)
        report = json.loads((dest / "report.json").read_text())
        if report["library_sha256"] != fingerprint:
            raise RuntimeError("all variants must use the same native binary")
        reports.append(str(dest / "report.json"))
        report_data.append(report)
        for row in report["runs"]:
            if not row["warmup"] and not row["parity"]:
                samples[variant].append(row["metrics"]["timings_seconds"]["denoise"])
        summary = {"samples": samples, "reports": reports, "library_sha256": fingerprint}
        summary["aggregate"] = summarize_reports(report_data)
        summary["conditions"] = {"size": a.size, "steps": a.steps, "seed": a.seed,
                                 "control": "production_default" if a.control_default else "unfused",
                                 "control_legacy_swiglu": a.control_legacy_swiglu,
                                 "prompt": a.prompt, "model": a.model,
                                 "experimental_mpp": a.mpp,
                                 "disable_mpp": a.no_mpp,
                                 "experimental_mpp_dual": a.mpp_dual,
                                 "experimental_mpp_projections": a.mpp_projections,
                                 "experimental_mpp_attention_out": a.mpp_attn_out,
                                 "experimental_mpp_qkv_prepare": a.mpp_qkv_prepare,
                                 "experimental_gate_norm": a.gate_norm,
                                 "cache_context": a.cache_context,
                                 "gate_norm_virtual_threads": a.gate_norm_virtual_threads,
                                 "experimental_vector_norm": a.norm_vector,
                                 "scalar_norm_override": a.scalar_norm,
                                 "norm_threads": a.norm_threads,
                                 "virtual_norm_threads": a.virtual_norm_threads}
        summary["system_state_assessment"] = assess_system_state(report_data,
                                                                  summary["conditions"])
        if all(samples.values()):
            summary["diffusion_speedup"] = statistics.median(samples["baseline"]) / statistics.median(samples["fused"])
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
