#!/usr/bin/env python3
"""Rank observed GGUF candidates at 6/8/10/16 GiB; never certify a RAM cap.

Timing and memory reports must be separate, successful, same-library runs with
the same workload/source/encoder and exact PNG identity. Warmups remain in the
raw evidence but are excluded from ranking. This is an empirical screen only;
unobserved driver/envelope upper bounds and the formal campaign remain unknown.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("missing/invalid content SHA-256")
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("invalid " + name)
    return value


def identity(report, row):
    request, metrics = row["request"], row["metrics"]
    if type(row.get("prefetch")) is not int or row["prefetch"] not in (-1, 0, 1, 2):
        raise ValueError("invalid prefetch")
    execution = request["execution"].get("quantized_execution")
    q = metrics.get("quantized_execution")
    native = execution is None
    importer = metrics.get("gguf_import")
    if native:
        if row["prefetch"] != -1 or report.get("native_packed_math_profile") != "z-mlx-compat-affine-v1":
            raise ValueError("unbound native packed math profile")
        execution = {"schema_version": 1, "enabled": True, "precision_profile": "z-mlx-compat-affine-v1"}
        dit_sha = sha(report.get("dit_source_sha256"))
        if report.get("native_import","mlx")=="cpu_direct":
            if (not importer or importer.get("recipe")!="gguf-mlx-compat-affine-packed-bank-v1" or
                    importer.get("allocator_cache_limit_bytes")!=0 or sha(importer.get("source_sha256"))!=dit_sha):
                raise ValueError("unbound CPU-direct packed import")
        elif importer:
            raise ValueError("unexpected native packed import recipe")
    else:
        if (not q or q["slot_count"] != row["prefetch"] + 1 or
                q["fill_count"] != 30 * request["sampling"]["steps"] or
                execution["prefetch_layers"] != row["prefetch"] or
                q["source_residency"] != execution["source_residency"]):
            raise ValueError("wrong observed DiT slots/fills")
        dit_sha = sha(q["source_sha256"])
    encoder = metrics.get("encoder_quantized_execution")
    if not encoder:
        raise ValueError("budget screen requires explicit GGUF encoder identity, not an unbound dense encoder")
    if report.get("prompt_cache_policy") == "miss" and metrics.get("prompt_cache_hit") is not False:
        raise ValueError("prompt-cache miss arm reused conditioning")
    if (report.get("prompt_cache_policy") == "hit" and row.get("measurement") == "timing" and
            not row.get("warmup", False) and metrics.get("prompt_cache_hit") is not True):
        raise ValueError("prompt-cache hit timing arm did not hit; explicit warmup required")
    if "dump_tensors" in request or row.get("cancellation_triggered"):
        raise ValueError("diagnostic/cancel request cannot be a performance arm")
    shared = {
        "library_sha256": sha(report["library_sha256"]),
        "loaded_libraries": {name: {"sha256": sha(value["sha256"])}
            for name, value in report["loaded_libraries"].items()},
        "dit_sha256": dit_sha,
        "encoder_sha256": sha(encoder["source_sha256"]),
        "encoder_layout_digest": sha(encoder["layout_digest"]),
        "encoder_profile": encoder["precision_profile"],
        "encoder_controls": {key: value for key, value in report.get("encoder_environment", {}).items()
            if key not in ("TURBOCIDER_Z_QWEN3_GGUF", "TURBOCIDER_Z_QWEN3_GGUF_CONFIG")},
        "prompt_cache_policy": report["prompt_cache_policy"],
        "math": {k: v for k, v in execution.items() if k not in ("prefetch_layers", "source_residency")},
        "workload": {k: v for k, v in request.items() if k not in ("outputs", "execution")},
        "outputs": [{k: v for k, v in output.items() if k != "path"} for output in request["outputs"]],
        "execution": {k: v for k, v in request["execution"].items() if k != "quantized_execution"},
        "png_sha256": sha(row["png_sha256"]),
    }
    candidate = ("native_packed:cpu_direct" if importer else "native_packed") if native else f'{execution["source_residency"]}:p{row["prefetch"]}'
    eval_policy=report.get("gpu_eval_policy","default")
    if eval_policy not in ("default","each-main-block-v1"): raise ValueError("unknown GPU eval policy")
    if eval_policy!="default": candidate+=":"+eval_policy
    return shared, candidate


def screen(reports, budgets=(6, 8, 10, 16), buffer_percent=10, min_samples=4, budget_unit="GiB"):
    if not budgets or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in budgets):
        raise ValueError("budgets must be positive integer GiB")
    if isinstance(buffer_percent, bool) or not isinstance(buffer_percent, int) or not 0 <= buffer_percent < 100:
        raise ValueError("invalid buffer percent")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("at least two screen samples required")
    if budget_unit not in ("GB", "GiB"):
        raise ValueError("budget unit must be GB or GiB")
    common, candidates = None, {}
    for report in reports:
        mode = report.get("measurement")
        if mode not in ("timing", "memory") or report.get("status") != "completed" or not report.get("runs"):
            raise ValueError("requires completed separate timing/memory reports")
        for row in report["runs"]:
            if row.get("status") != 0 or row.get("error") or not row.get("metrics") or row.get("measurement") != mode:
                raise ValueError("failed or mislabeled request")
            if type(row.get("warmup", False)) is not bool:
                raise ValueError("invalid warmup marker")
            shared, name = identity(report, row)
            if common is None:
                common = shared
            if common != shared:
                raise ValueError("workload/library/component/profile/PNG identity changed")
            record = candidates.setdefault(name, {"timings_seconds": [], "observed_peak_bytes": 0,
                "memory_sample_max_gap_seconds": 0, "managed_weights_peak_bytes": 0,
                "memory_runs": 0, "sampling_inconclusive": False, "swap_observed": False})
            if name.startswith("native_packed:cpu_direct"):
                import_metrics=row["metrics"]["gguf_import"]
                consumer=import_metrics.get("consumer_revision","initial-retained-bank-v1")
                if "consumer_revision" in record and record["consumer_revision"]!=consumer:
                    raise ValueError("candidate import consumer changed between measurements")
                record["consumer_revision"]=consumer
                layout = sha(import_metrics["plan_digest"])
                if "layout_digest" in record and record["layout_digest"] != layout:
                    raise ValueError("candidate import plan changed between measurements")
                record["layout_digest"] = layout
            elif not name.startswith("native_packed"):
                layout = sha(row["metrics"]["quantized_execution"]["layout_digest"])
                if "layout_digest" in record and record["layout_digest"] != layout:
                    raise ValueError("candidate layout changed between measurements")
                record["layout_digest"] = layout
            if row.get("warmup", False):
                continue
            if mode == "timing":
                if row.get("memory_samples"):
                    raise ValueError("timing arm has a memory observer")
                record["timings_seconds"].append(positive(row["wall_seconds"], "request timing"))
            else:
                samples = row.get("memory_samples")
                if not samples or row.get("sampling_errors"):
                    raise ValueError("missing/failed memory observer")
                gap = positive(row["memory_sample_max_gap_seconds"], "memory sample gap")
                record["sampling_inconclusive"] |= gap > .05
                record["memory_sample_max_gap_seconds"] = max(record["memory_sample_max_gap_seconds"], gap)
                # Kernel-reported process lifetime peak is conservative across
                # same-process arms; never subtract an earlier/larger peak.
                peak = max(positive(s[key], "process footprint") for s in samples
                    for key in ("phys_footprint_bytes", "lifetime_max_phys_footprint_bytes"))
                record["observed_peak_bytes"] = max(record["observed_peak_bytes"], peak)
                q = row["metrics"].get("quantized_execution") or row["metrics"].get("gguf_import")
                e = row["metrics"]["encoder_quantized_execution"]
                if not q:
                    record["native_dit_managed_weights"] = "unknown; only process footprint observed"
                record["managed_weights_peak_bytes"] = max(record["managed_weights_peak_bytes"],
                    positive(q["managed_peak_bytes"], "DiT managed peak") if q else 0,
                    positive(e["managed_peak_bytes"], "encoder managed peak"))
                if "Swapouts" not in row.get("vm_deltas", {}):
                    raise ValueError("missing swap observation")
                record["swap_observed"] |= row["vm_deltas"]["Swapouts"] != 0
                record["memory_runs"] += 1
    if common is None:
        raise ValueError("empty reports")
    for name, record in candidates.items():
        if len(record["timings_seconds"]) < min_samples or not record["memory_runs"]:
            raise ValueError("insufficient timing or memory evidence: " + name)
        record["warm_median_seconds"] = statistics.median(record["timings_seconds"])
    result = {"schema": "tc-gguf-memory-budget-screen-v1", "production_qualified": False,
        "scope": "fastest observed same-source packed/bounded candidate that fits sampled footprint; NOT enforced whole-request caps or true small-device qualification",
        "whole_request_memory": "unknown", "formal_performance": "not_run", "identity": common,
        "buffer_percent": buffer_percent, "budget_unit": budget_unit, "candidates": candidates, "budgets": []}
    for value in budgets:
        limit = value << 30 if budget_unit == "GiB" else value * 1_000_000_000
        available = limit * (100 - buffer_percent) // 100
        fitting = [name for name, record in candidates.items()
            if max(record["observed_peak_bytes"], record["managed_weights_peak_bytes"]) <= available
            and not record["swap_observed"] and not record["sampling_inconclusive"]]
        best = min(fitting, key=lambda name: (candidates[name]["warm_median_seconds"], name)) if fitting else None
        result["budgets"].append({"budget_value": value, "budget_unit": budget_unit,
            "budget_gib" if budget_unit == "GiB" else "budget_gb": value,
            "limit_bytes": limit, "screen_available_bytes": available,
            "observed_fitting_candidates": sorted(fitting), "fastest_observed_candidate": best,
            "status": "observed_fit_only" if best else "no_supported_observation",
            "bounded_certified": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", "--budgets-gib", dest="budgets", type=int, nargs="+", default=[6, 8, 10, 16])
    parser.add_argument("--budget-unit", choices=["GB", "GiB"], default="GiB")
    parser.add_argument("--buffer-percent", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink(): parser.error("output already exists")
    result = screen([json.loads(path.read_text()) for path in args.reports], args.budgets,
                    args.buffer_percent, args.min_samples, args.budget_unit)
    result["raw_reports"] = [{"id": path.parent.name, "sha256": digest(path)} for path in args.reports]
    result["verifier_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    for row in result["budgets"]:
        print(row["budget_value"], row["budget_unit"], row["fastest_observed_candidate"], row["status"])


if __name__ == "__main__": main()
