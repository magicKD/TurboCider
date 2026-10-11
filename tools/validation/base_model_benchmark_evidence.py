#!/usr/bin/env python3
"""Export dated campaign samples without reloading or rehashing model weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def measured(route, cold, warm, **details):
    if not cold or not warm or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                                  for value in (*cold, *warm)):
        raise ValueError("finite positive complete-request samples required")
    return dict(route=route, cold_seconds=cold, warm_seconds=warm,
                warm_sample_count=len(warm), warm_median_seconds=statistics.median(warm), **details)


def source(path, statuses):
    raw = (ROOT / path).read_bytes()
    data = json.loads(raw)
    if data.get("status") not in statuses:
        raise ValueError("incomplete evidence: " + path)
    return data, dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), status=data["status"])


def runtime_campaign(path, model, size, steps, note):
    data, binding = source(path, ("complete",))
    if data["model"] != model or data["size"] != size or data["steps"] != steps or data.get("lora"):
        raise ValueError("wrong base checkpoint/task")
    routes = []
    for route in sorted(set(data["routes"])):
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        routes.append(measured(route, [trial["wall_seconds"][0] for trial in trials],
            [value for trial in trials for value in trial["wall_seconds"][1:]],
            warm_denoise_seconds=[value for trial in trials for value in trial["denoise_seconds"][1:]]))
    return dict(evidence=binding, model=model, geometry=[size, size], steps=steps,
                library_sha256=data["library_sha256"], routes=routes, note=note,
                timing_scope=data["timing_scope"])


def export():
    campaigns = []
    data, binding = source("docs/design/validation/qwen21-conditioned-reference-20261010.json",
                           ("complete_diagnostic",))
    routes = []
    for trial in data["trials"]:
        target = trial["target"]
        if trial["mode"] == "reference":
            rows = target["requests"]
            routes.append(measured("original-unsloth-gpu", [rows[0]["generation_seconds"]],
                [row["generation_seconds"] for row in rows[1:]], cold_encoder_seconds=[rows[0]["condition_seconds"]],
                warm_encoder_seconds=[row["condition_seconds"] for row in rows[1:]],
                warm_denoise_seconds=[row["sampling_seconds"] for row in rows[1:]],
                warm_http_seconds=[row["http_request_seconds"] for row in rows[1:]],
                timing_scope="reference generate_image span; HTTP measured separately"))
        else:
            rows = target["timings"]
            routes.append(measured(trial["mode"], [rows[0]["request_wall"]],
                [row["request_wall"] for row in rows[1:]], cold_encoder_seconds=[rows[0]["text_encode"]],
                warm_encoder_seconds=[row["text_encode"] for row in rows[1:]],
                warm_denoise_seconds=[row["denoise"] for row in rows[1:]],
                timing_scope="native request_wall including VAE/export"))
    campaigns.append(dict(evidence=binding, model="qwen-image-2.1-original-Q4_K_M", geometry=[512, 512],
        steps=40, date="2026-10-10", routes=routes, source_sha256=data["source_sha256"],
        note="each target preceded by actual BF16 streaming 40 steps x3; two fresh warm prompts; not identical OS page state"))

    data, binding = source("outputs/qwen21-read-ahead-v1-matched-40step-20261010/summary.json",
                           ("complete_diagnostic",))
    routes = []
    for trial in data["trials"]:
        rows = trial["timings"]
        memory = trial["memory"]["verified"]
        if memory["complete"] is not True:
            raise ValueError("incomplete matched-memory evidence")
        routes.append(measured(trial["mode"], [rows[0]["request_wall"]],
            [row["request_wall"] for row in rows[1:]],
            warm_encoder_seconds=[row["text_encode"] for row in rows[1:]],
            warm_denoise_seconds=[row["denoise"] for row in rows[1:]],
            memory={key: memory[key] for key in ("tree_peak_phys_footprint_bytes", "compression_bytes",
                                                "swap_in_bytes", "swap_out_bytes")}))
    campaigns.append(dict(evidence=binding, model="qwen-image-2.1-format-and-residency-comparison",
        geometry=[512, 512], steps=40, date="2026-10-10", routes=routes,
        timing_scope="native request_wall including VAE/export; one cold and two fresh warm per route",
        observed_memory_within_ten_percent=data["observed_memory_within_ten_percent"],
        note="physical peak is process-tree load+cold+warm+exit; not a hard RAM cap or equal-precision comparison"))

    data, binding = source("docs/design/validation/local512-qwen-frozen-base-20261010.json",
                           ("complete_busy_diagnostic",))
    routes = []
    for route in ("gpu", "all", "frozen"):
        trials = [trial for experiment in data["experiments"] for trial in experiment["trials"]
                  if trial["mode"] == route]
        routes.append(measured(route, [trial["cold_request_seconds"] for trial in trials],
                               [value for trial in trials for value in trial["warm_request_seconds"]]))
    campaigns.append(dict(evidence=binding, model="qwen-image-2.1-BF16", geometry=[512, 512], steps=40,
        date="2026-10-10", routes=routes, library_sha256=data["build"]["private_library_sha256"],
        timing_scope="native request_wall including VAE/export; request-local encoder on every route",
        note="forward/reverse; both CPU load checks failed; busy diagnostic, not idle/default qualification"))

    runtime_specs = (
        ("outputs/runtime-ane/qwen-v1-base-40step-forward-875b/summary.json", "qwen-image-2.1", 1024, 40,
         "forward campaign; combine raw warm samples with reverse, not route medians"),
        ("outputs/runtime-ane/qwen-v1-base-40step-reverse-retry-875b/summary.json", "qwen-image-2.1", 1024, 40,
         "reverse campaign; README quotes the pooled actual warm samples across both directions"),
        ("outputs/runtime-ane/z-base-v2-current-palindrome/summary.json", "z-image-turbo", 512, 8,
         "September retained build; GPU/runtime v2/frozen palindromic order; single fox prompt"),
        ("outputs/runtime-ane/z-base-1024-v1-gpu-abba-875b/summary.json", "z-image-turbo", 1024, 8,
         "2026-09-30 lighthouse; GPU/runtime v1 ABBA"),
        ("outputs/runtime-ane/z-base-1024-v1-portrait-abba-875b/summary.json", "z-image-turbo", 1024, 8,
         "2026-09-30 portrait; separate prompt campaign"),
        ("outputs/runtime-ane/z-gguf-q8-1024-v1-abba-875b/summary.json", "z-image-turbo-gguf", 1024, 8,
         "2026-09-30 Q8 lighthouse; GPU/runtime v1 ABBA; not BF16 checkpoint"),
        ("outputs/runtime-ane/z-gguf-q8-1024-v1-portrait-abba-875b/summary.json", "z-image-turbo-gguf", 1024, 8,
         "2026-09-30 Q8 portrait; runtime PNG differs between forward/reverse"),
    )
    for spec in runtime_specs:
        campaigns.append(runtime_campaign(*spec))
    ltx_path = "docs/design/validation/ltx-ane-and-720p-tiled-2026-09-13.json"
    raw = (ROOT / ltx_path).read_bytes()
    ltx = json.loads(raw)
    routes = [dict(route=name, warm_sample_count=1, warm_seconds=[ltx[key]["warm_request_seconds"]],
                   warm_median_seconds=ltx[key]["warm_request_seconds"],
                   quality_status="baseline" if name == "gpu" else "experimental-quality-fail")
              for name, key in (("gpu", "gpu_baseline"), ("ane-int8", "ane_int8"), ("ane-fp16", "ane_fp16"))]
    campaigns.append(dict(evidence=dict(path=ltx_path, sha256=hashlib.sha256(raw).hexdigest(),
        status="retained-structured-validation"), model="ltx-2.5-distilled", geometry=[704, 448, 97], steps=11,
        date="2026-09-13", routes=routes, timing_scope="warm request wall; single sample",
        note="INT8 small latency win failed quality; FP16 slower and failed quality; not automatically selected"))
    flux, binding = source("docs/design/validation/flux9-bf16-framework-20261011.json", ("complete_diagnostic",))
    routes = []
    for engine in ("native", "mflux"):
        median = flux["medians"][engine]
        routes.append(measured(engine, median["cold_seconds"], median["samples"],
                               repeated_prompt_seconds=median["repeated_prompt_seconds"],
                               constructor_seconds=median["constructor_seconds"]))
    campaigns.append(dict(evidence=binding, model="flux2-klein-9b-BF16", geometry=[512, 512], steps=4,
        date="2026-10-11", routes=routes, timing_scope=flux["timing_scope"],
        note="same BF16 files; native/mflux ABBA; compression/swap retained; mlx-serve rejects original key/layout, no latency denominator",
        unsupported_same_layout_frameworks=["mlx-serve26.10.1"],
        native_hybrid_status="9B DiT partition unvalidated; no automatic/claimed mixed benchmark"))
    z, binding = source("docs/design/validation/z-base-formats-20261011.json", ("complete_diagnostic",))
    for format_name in ("bf16", "nvfp4", "convrot", "q4", "q8"):
        routes = []
        for route in z["routes"]:
            if route["format"] == format_name:
                routes.append(measured(route["name"], route["cold_seconds"], route["fresh_warm_seconds"],
                    repeated_prompt_seconds=route["repeat_seconds"], physical_peaks=route["physical_peaks"],
                    route_kind=route["mode"]))
        campaigns.append(dict(evidence=binding, model="z-image-turbo-local-" + format_name, geometry=[512, 512],
            steps=8, date="2026-10-11", routes=routes,
            timing_scope="native request_wall; four fresh warm per config across full forward/reverse; repeat excluded",
            note="same library/original BF16 encoder/VAE/tokenizer; source representation and opt-in arithmetic differ; all fallback/retry/memory pressure retained",
            source_library_inputs_matched=z["native_inputs_matched"]))
    current_ltx, binding = source("docs/design/validation/ltx-base-current-20261011.json", ("complete_diagnostic",))
    routes = [measured(route["mode"], route["cold_seconds"], route["warm_seconds"],
                       physical_peaks=route["physical_peaks"]) for route in current_ltx["routes"]]
    campaigns.append(dict(evidence=binding, model="ltx-2.5-distilled-current-full-media", geometry=[704, 448, 97],
        steps=11, date="2026-10-11", routes=routes,
        timing_scope="native full MP4 request_wall; one cold and one immediate conditioning-cache-hit warm per process",
        note="GPU/INT8 forward+reverse two warm each; FP16 one warm; media drift and system pressure retained; stock mlx-serve original-pack preflight rejects before generation"))
    return dict(schema="tc-local-base-benchmark-samples-v1", date="2026-10-11", status="partial_expanded_evidence_catalog",
                hardware="Apple M4 Max 64 GiB; dated cohorts retain original build/OS settings",
                qualification_passed=False, goal_complete=False,
                completion_audit="docs/benchmark/COMPLETION_AUDIT.md",
                goal_completion_scope="expanded local family/precision/parameter and best GPU/ANE/framework comparisons remain active; prior-stage completion is not expanded-goal proof; not release qualification",
                expansion_status="docs/benchmark/EXPANSION_STATUS.md",
                campaigns=campaigns,
                missing_current_comparisons=["Qwen BF16/mflux full40 comparison and affine4/8 routes",
                    "Z original BF16/mflux and additional affine4/8 framework routes", "FLUX9B quantized framework and native hybrid qualification",
                    "LTX best full-media native configurations and runnable reference quantized layouts", "cross-family parameter screening and durable actually-loadable best configurations"],
                unmeasured_or_inapplicable=["current FastH3 INT6 DiT and original H3 video VAE assets absent",
                    "FLUX9B DiT hybrid partition unvalidated", "NVFP4 runtime GPU-only", "mlx-serve original FLUX layout rejected and original LTX stock memory preflight rejected",
                    "additional converted precision/model packs are not equal-source original-format benchmark denominators"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh evidence output required")
    result = export()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False); target.write("\n")
    print(json.dumps(dict(status=result["status"], campaigns=len(result["campaigns"]))))


if __name__ == "__main__":
    main()
