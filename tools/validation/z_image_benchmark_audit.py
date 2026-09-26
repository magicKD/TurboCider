"""Read-only audit of a completed Z-Image ABBA with exact file parity.

Uses only the standard library. This verifies matched benchmark conditions
and byte equality, not perceptual quality or numerical finiteness of tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    reports = [json.loads(Path(p).read_text()) for p in summary["reports"]]
    require([r["variant"] for r in reports] == ["baseline", "fused", "fused", "baseline"],
            "incomplete or non-ABBA schedule")
    expected_library = summary["library_sha256"]
    libraries = reports[0]["runtime_libraries"]
    require("libturbocider.dylib" in libraries and "libmlx.dylib" in libraries,
            "missing actual loaded library fingerprints")
    require(libraries["libturbocider.dylib"]["sha256"] == expected_library,
            "loaded native library differs from requested fingerprint")
    request_reference = None
    parity_reference = None
    conditions = summary["conditions"]
    require(conditions.get("control") in ("unfused", "production_default"),
            "unknown control semantics")
    samples = {"baseline": [], "fused": []}
    walls = {"baseline": [], "fused": []}
    variant_environments = {}
    boot_sessions = [report.get("system_state", {}).get("boot_time") for report in reports]
    if any(value is not None for value in boot_sessions):
        require(all(value is not None for value in boot_sessions),
                "incomplete boot-session metadata")
        require(len(set(boot_sessions)) == 1, "benchmark crossed boot sessions")
    for report in reports:
        variant = report["variant"]
        require(report["library_sha256"] == expected_library, "native library changed")
        require(report["runtime_libraries"] == libraries, "loaded runtimes differ")
        require(report["hardware"] == reports[0]["hardware"], "hardware metadata differs")
        environment = report["environment"]
        if conditions["control"] == "unfused":
            disable_flags = ("TURBOCIDER_Z_DISABLE_FUSED_QKV",
                             "TURBOCIDER_Z_DISABLE_FUSED_MOD",
                             "TURBOCIDER_Z_DISABLE_MPP_SWIGLU",
                             "TURBOCIDER_Z_DISABLE_MPP_PROJECTIONS",
                             "TURBOCIDER_Z_DISABLE_MPP_QKV_PREPARE",
                             "TURBOCIDER_Z_DISABLE_GATE_NORM",
                             "TURBOCIDER_Z_DISABLE_CACHE_CONTEXT")
            if variant == "baseline":
                require(all(environment.get(k) == "1" for k in disable_flags),
                        "unfused control switches missing")
            else:
                require(not any(k in environment for k in disable_flags),
                        "default candidate has disable switches")
        require(environment == variant_environments.setdefault(variant, environment),
                "same-arm environments differ")
        rows = report["runs"]
        require(len(rows) >= 3 and rows[0]["warmup"] and not rows[0]["parity"],
                "missing first cold request")
        require(sum(bool(r["warmup"]) for r in rows) == 1, "wrong cold sample count")
        quality = [r for r in rows if r["parity"]]
        require(len(quality) == 1 and rows[-1]["parity"], "missing final parity request")
        timed = [r for r in rows if not r["warmup"] and not r["parity"]]
        require(timed, "missing warm samples")
        for row in rows:
            require(row["error"] is None and row["metrics"], "generation failed")
            request = row["request"]
            comparable = {k: v for k, v in request.items() if k not in ("output", "dump_tensors")}
            if request_reference is None:
                request_reference = comparable
            require(comparable == request_reference, "requests differ")
            require(request["width"] == conditions["size"] and request["height"] == conditions["size"]
                    and request["steps"] == conditions["steps"] and request["seed"] == conditions["seed"]
                    and request["prompt"] == conditions["prompt"], "summary workload mismatch")
            metrics = row["metrics"]
            require(metrics["actual_denoise_steps"] == request["steps"], "actual step mismatch")
            plan = metrics["plan"]
            require(request["execution"] == "gpu" and plan["requested_execution"] == "gpu"
                    and not metrics.get("hybrid"), "not pure GPU")
            require(plan["precision"] == "bf16" and not plan.get("quantized_cache"),
                    "not the qualified unquantized BF16 path")
            require(plan["requested_shape"][:2] == [conditions["size"], conditions["size"]],
                    "plan shape mismatch")
        for row in timed:
            seconds = row["metrics"]["timings_seconds"]["denoise"]
            wall = row["wall_seconds"]
            require(0 < seconds <= wall < float("inf"), "invalid timing")
            samples[variant].append(seconds)
            walls[variant].append(wall)
        quality_request = quality[0]["request"]
        names = {"z_latent_initial.safetensors", "z_latent_final.safetensors", "z_decoded.safetensors"}
        names.update(f"z_latent_step_{i+1}.safetensors" for i in range(conditions["steps"]))
        folder = Path(quality_request["dump_tensors"])
        require({p.name for p in folder.glob("*.safetensors")} == names,
                "missing or unexpected parity tensor files")
        hashes = {name: digest(folder / name) for name in sorted(names)}
        hashes["png"] = digest(quality_request["output"])
        if parity_reference is None:
            parity_reference = hashes
        require(hashes == parity_reference, "parity files differ")
    require(samples == summary["samples"], "summary samples differ from source reports")
    medians = {k: statistics.median(v) for k, v in samples.items()}
    wall_medians = {k: statistics.median(v) for k, v in walls.items()}
    assessment = summary.get("system_state_assessment")
    if assessment is not None:
        require(assessment.get("status") != "invalid_for_performance",
                "system state outside qualified performance envelope")
    # Older reports predate explicit system-state metadata. Still reject the
    # known degraded M4 Max scale so a legacy JSON cannot silently re-enter
    # performance analysis merely because its ABBA/parity accounting is valid.
    if reports[0]["hardware"].get("gpu") == "Apple M4 Max":
        limits = {(512, 8): (9.0, 1.0), (1024, 9): (45.0, 2.0)}
        limit = limits.get((conditions["size"], conditions["steps"]))
        if limit:
            reference = "baseline" if samples["baseline"] else "fused"
            reference_reports = [report for report in reports if report["variant"] == reference]
            vae_samples = [row["metrics"]["timings_seconds"]["vae_decode"]
                           for report in reference_reports for row in report["runs"]
                           if not row["warmup"] and not row["parity"]]
            require(medians[reference] <= limit[0] and
                    statistics.median(vae_samples) <= limit[1],
                    "system state outside qualified performance envelope")
    return {"passed": True, "scope": "matched ABBA and exact file parity, not perceptual quality",
            "warm_samples": {k: len(v) for k, v in samples.items()},
            "library_sha256": expected_library, "environments": variant_environments,
            "diffusion_medians": medians, "wall_medians": wall_medians,
            "diffusion_speedup": medians["baseline"] / medians["fused"],
            "wall_speedup": wall_medians["baseline"] / wall_medians["fused"],
            "parity_hashes": parity_reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.summary), indent=2))


if __name__ == "__main__":
    main()
