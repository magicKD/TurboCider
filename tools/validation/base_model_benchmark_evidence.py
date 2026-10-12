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


def qwen_cache_campaign(path, cache_mib):
    data, binding = source(path, ("complete_diagnostic",))
    order = ["native-gpu", "native-frozen", "mflux-bf16", "mflux-bf16", "native-frozen", "native-gpu"]
    if (cache_mib not in (1024, 4096) or data["reference_cache_mib"] != cache_mib or
            data["steps"] != 40 or data["requests_per_process"] != 4 or data["order"] != order or
            [trial["index"] for trial in data["trials"]] != list(range(6)) or
            [trial["route"] for trial in data["trials"]] != order or
            data["reference_latency_ranking_valid"] is not True):
        raise ValueError("complete valid same-window Qwen cache candidate required")
    routes = []
    for row in data["routes"]:
        trials = [trial for trial in data["trials"] if trial["route"] == row["route"]]
        if len(trials) != 2 or any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete cache candidate memory required")
        if len(row["fresh_warm_seconds"]) != 4 or len(row["cold_seconds"]) != 2 or len(row["repeat_seconds"]) != 2:
            raise ValueError("four fresh samples and separate cold/repeat required")
        if row["route"] == "mflux-bf16" and any(
                trial["report"]["reference_cache_limit_bytes"] != cache_mib << 20 for trial in trials):
            raise ValueError("actual reference cache limit differs")
        routes.append(measured(row["route"], row["cold_seconds"], row["fresh_warm_seconds"],
            repeated_prompt_seconds=row["repeat_seconds"], constructor_seconds=row["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials],
            compression_bytes=[trial["memory"]["compression_bytes"] for trial in trials],
            swap_out_bytes=[trial["memory"]["swap_out_bytes"] for trial in trials]))
    if {row["route"] for row in routes} != set(order) or len(routes) != 3:
        raise ValueError("all three matched cache candidate controls required")
    return dict(evidence=binding, model=f"qwen-image-2.1-cache{cache_mib}-framework-cohort",
        geometry=[512, 512], steps=40, date=data["date"], routes=routes,
        reference_cache_mib=cache_mib, reference_comparison_valid=True, timing_scope=data["timing_scope"],
        note="complete six-process same-window GPU/frozen/mflux BF16 cache candidate; four fresh/route,24PNG/6memory; actual process-local limit checked, cold/repeat/pressure retained;1024/4096 medians differ only0.1%, no causal cache-speed ranking or pooling with stock/cache0")


def z_upstream_campaign(path):
    data, binding = source(path, ("complete_diagnostic",))
    order = ["native-gpu", "native-frozen", "mflux-bf16", "mflux-fp16", "mflux-fp16", "mflux-bf16", "native-frozen", "native-gpu"]
    if (data["steps"] != 8 or data["requests_per_process"] != 4 or data["order"] != order or
            len(data["trials"]) != 8 or data["mflux_package_version"] != "0.22.0" or
            data["reference_cache_mib"] != 1024 or data["reference_latency_ranking_valid"] is not True or
            len(data["reference_compute_quality"]) != 8):
        raise ValueError("complete source-bound current Z BF16/float16 candidate required")
    routes = []
    for row in data["routes"]:
        trials = [trial for trial in data["trials"] if trial["route"] == row["route"]]
        if len(trials) != 2 or len(row["fresh_warm_seconds"]) != 4 or any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete current Z samples and process-tree memory required")
        if row["route"] == "mflux-bf16" and any(trial["report"].get("compute_precision_name") is not None for trial in trials):
            raise ValueError("current BF16 control cannot enable float16 math")
        if row["route"] == "mflux-fp16" and any(
                trial["report"]["compute_precision_name"] != "float16" or
                trial["report"]["transformer_floating_parameter_dtypes"].get("mlx.core.float16", 0) <= 0 or
                any(run["generation_parameters"].get("compute_precision") != "float16" for run in trial["report"]["runs"])
                for trial in trials):
            raise ValueError("current Z float16 needs actual parameters and generation metadata")
        routes.append(measured(row["route"], row["cold_seconds"], row["fresh_warm_seconds"],
            repeated_prompt_seconds=row["repeat_seconds"], constructor_seconds=row["constructor_seconds"],
            physical_peaks=row["physical_peaks"],
            compression_bytes=[trial["memory"]["compression_bytes"] for trial in trials],
            swap_out_bytes=[trial["memory"]["swap_out_bytes"] for trial in trials]))
    return dict(evidence=binding, model="z-image-turbo-mflux022-bf16-fp16-cohort", geometry=[512, 512],
        steps=8, date=data["date"], routes=routes, mflux_commit=data["mflux_commit"],
        mflux_package_version=data["mflux_package_version"], reference_cache_mib=1024,
        timing_scope=data["timing_scope"], reference_comparison_valid=True,
        note="current upstream0.22.0 isolated declared-dependency environment,8process/32PNG/8memory,4fresh/route; sourceBF16 stock7.976479s versus partialfloat16 8.355058s is4.746% slower, nativeGPU7.354224/frozen5.959064s; native speed wins but reference peak22.51GB is lower than native31.81-32.16GB; old reference9.519604s is a separate cohort/environment, no pooling or single-factor release causality")


def qwen_upstream_campaign(path):
    data, binding = source(path, ("complete_diagnostic",))
    order = ["native-gpu", "native-frozen", "mflux-bf16", "mflux-fp16", "mflux-fp16", "mflux-bf16", "native-frozen", "native-gpu"]
    if (data["steps"] != 40 or data["requests_per_process"] != 4 or data["order"] != order or
            len(data["trials"]) != 8 or data["mflux_package_version"] != "0.22.0" or
            data["reference_cache_mib"] != 1024 or data["reference_latency_ranking_valid"] is not True or
            len(data["reference_compute_quality"]) != 8):
        raise ValueError("complete original40-step current Qwen source-bound cohort required")
    routes = []
    for row in data["routes"]:
        trials = [trial for trial in data["trials"] if trial["route"] == row["route"]]
        if len(trials) != 2 or len(row["fresh_warm_seconds"]) != 4 or any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("all four current Qwen fresh samples and complete memory required")
        if row["route"].startswith("mflux-") and any(trial["report"]["reference_cache_limit_bytes"] != 1024 << 20 for trial in trials):
            raise ValueError("actual current Qwen reference cache differs")
        if row["route"] == "mflux-fp16" and any(
                trial["report"]["compute_precision_name"] != "float16" or
                trial["report"]["transformer_floating_parameter_dtypes"].get("mlx.core.float16", 0) <= 0 or
                any(run["generation_parameters"].get("compute_precision") != "float16" for run in trial["report"]["runs"])
                for trial in trials):
            raise ValueError("current Qwen float16 requires actual parameters and outputs")
        routes.append(measured(row["route"], row["cold_seconds"], row["fresh_warm_seconds"],
            repeated_prompt_seconds=row["repeat_seconds"], constructor_seconds=row["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials],
            compression_bytes=[trial["memory"]["compression_bytes"] for trial in trials],
            swap_out_bytes=[trial["memory"]["swap_out_bytes"] for trial in trials]))
    return dict(evidence=binding, model="qwen-image-2.1-mflux022-bf16-fp16-cohort", geometry=[512, 512],
        steps=40, date=data["date"], routes=routes, mflux_package_version=data["mflux_package_version"],
        mflux_commit=data["mflux_commit"], reference_cache_mib=1024, reference_comparison_valid=True,
        timing_scope=data["timing_scope"],
        note="current0.22.0 independent declared-dependency environment,8process/32PNG/8memory,4fresh/route; referenceBF16 41.952435s beats nativeGPU43.308977s, frozen31.809596s remains faster; referencepartialFP16 42.000398s differs only0.114%, no stable compute-option win; nativeGPU peak29.77-30.47GB lower than reference34.51-34.93GB/frozen36.03-36.08GB; old reference45.450516s retained separately, no pooling or global dominance")


def flux_upstream_campaign(path):
    data, binding = source(path, ("complete_diagnostic",))
    routes = ("native-bf16", "native-hybrid", "native-q4", "native-predecoded", "mflux-bf16", "mflux-fp16", "mflux-q4")
    if (data["order"] != [*routes, *reversed(routes)] or len(data["trials"]) != 14 or
            data["mflux_package_version"] != "0.22.0" or data["reference_cache_mib"] != 0 or
            len(data["quality"]) != 64 or len(data["output_checks"]) != 7 or data["reference_comparison_valid"] is not True):
        raise ValueError("complete current seven-route FLUX source/precision cohort required")
    result = []
    for route in routes:
        values = data["medians"][route]
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        if len(values["samples"]) != 4 or len(trials) != 2 or any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete current FLUX fresh samples/memory required")
        if route == "native-hybrid" and any(
                row["metrics"]["hybrid"]["runtime_calls_session_total"] != 96 * (row["index"] + 1)
                for trial in trials for row in trial["report"]["runs"]):
            raise ValueError("actual complete96 current FLUX hybrid calls/request required")
        if route == "native-predecoded" and any(
                row["actual_compiled_single_blocks"] != 96 or
                row["native_event_counts"].get("affine_predecode_projection", 0) != (153 if row["index"] == 0 else 0)
                for trial in trials for row in trial["report"]["runs"]):
            raise ValueError("actual one-time153 current predecode and96 compiled blocks required")
        result.append(measured(route, values["cold_seconds"], values["samples"],
            repeated_prompt_seconds=values["repeated_prompt_seconds"], constructor_seconds=values["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials],
            compression_bytes=[trial["memory"]["compression_bytes"] for trial in trials],
            swap_out_bytes=[trial["memory"]["swap_out_bytes"] for trial in trials]))
    return dict(evidence=binding, model="flux2-klein-9b-mflux022-seven-route-cohort", geometry=[512, 512],
        steps=4, date=data["date"], routes=result, mflux_package_version=data["mflux_package_version"],
        mflux_commit=data["mflux_commit"], reference_cache_mib=0, reference_comparison_valid=True,
        timing_scope=data["timing_scope"],
        note="current0.22 isolated environment,14process/56PNG/14memory,4fresh/route; originalBF16 nativecompiled7.661436/hybrid7.433898/reference7.318347s, reference wins; hybrid saves2.970% aggregate but prompt-variable and cold21.69-21.86s; packednativeQ4 8.015237/referenceQ4 7.875917s, reference wins; nativepredecoded7.253946s saves9.498% versus ownpacked with peak24.36-24.43GB rather than11.92GB, sourceQ4 not strictBF16; referencepartialFP16 7.649527s slower4.525%; all negative results/cold/pressure and extra precision drift retained")


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
    native_qwen, binding = source("docs/design/validation/qwen21-native-cohort-current-20261011.json", ("complete_native_cohort",))
    if native_qwen["reference_latency_ranking_valid"] is not False:
        raise ValueError("invalid-reference diagnostic cannot become a speed denominator")
    routes = [measured(route["route"], route["cold_seconds"], route["fresh_warm_seconds"],
                       physical_peaks=route["physical_peaks"], repeated_prompt_seconds=route["repeat_seconds"])
              for route in native_qwen["routes"]]
    campaigns.append(dict(evidence=binding, model="qwen-image-2.1-current-BF16-native", geometry=[512, 512], steps=40,
        date="2026-10-11", routes=routes, timing_scope="native/API full PNG wall; four actual fresh warm per route; separate repeat; request-local GPU encoder",
        note="native GPU/frozen complete ABBA; reference omitted64 FFN weights and rendered noise, never promoted as a latency denominator"))
    for path, model, steps, note in (
        ("docs/design/validation/qwen21-valid-framework-current-20261011.json", "qwen-image-2.1-valid-framework-cohort", 40,
         "corrected complete source coverage; five-route ten-process palindrome; stock reference memory pressure and cross-engine image drift retained; not pure-kernel or strict BF16/W8A8 equivalence"),
        ("docs/design/validation/z-valid-framework-current-20261011.json", "z-image-turbo-valid-framework-cohort", 8,
         "seven-route fourteen-process palindrome; frozen/Private/Public and stock BF16/affine4/8; actual private4096/f32/retry retained; cold setup separate"),
        ("docs/design/validation/qwen21-cache0-framework-current-20261011.json", "qwen-image-2.1-cache0-framework-cohort", 40,
         "six-process same-window native GPU/frozen and mflux BF16 free-cache0 candidate; stock model/math unchanged; lower reference pressure, no pooling with stock cohort"),
        ("docs/design/validation/flux9-stock-precisions-current-20261011.json", "flux2-klein-9b-stock-reference-precisions", 4,
         "stock BF16/Q4/Q8 six-process palindrome; latest reference is faster than old BF16 ABBA; no native same-cohort denominator, no cross-cohort native ranking"),
    ):
        data, binding = source(path, ("complete_diagnostic",))
        if data["steps"] != steps or data["requests_per_process"] != 4 or data["order"] != list(reversed(data["order"])):
            raise ValueError("complete original-step fresh/repeat palindrome required")
        routes = [measured(route["route"], route["cold_seconds"], route["fresh_warm_seconds"],
            repeated_prompt_seconds=route["repeat_seconds"], constructor_seconds=route["constructor_seconds"])
            for route in data["routes"]]
        campaigns.append(dict(evidence=binding, model=model, geometry=[512, 512], steps=steps, date=data["date"],
            routes=routes, timing_scope=data["timing_scope"], note=note,
            reference_comparison_valid=data.get("reference_latency_ranking_valid", False),
            native_comparison_valid=data.get("native_latency_ranking_valid", True)))
    for cache_mib in (1024, 4096):
        campaigns.append(qwen_cache_campaign(
            f"docs/design/validation/qwen21-cache{cache_mib}-framework-current-20261012.json", cache_mib))
    campaigns.append(z_upstream_campaign("docs/design/validation/z-mflux022-bf16-fp16-current-20261012.json"))
    campaigns.append(qwen_upstream_campaign("docs/design/validation/qwen21-mflux022-bf16-fp16-current-20261012.json"))
    campaigns.append(flux_upstream_campaign("docs/design/validation/flux9-mflux022-current-20261012.json"))
    flux, binding = source("docs/design/validation/flux9-bf16-current-v5-20261011.json", ("complete_diagnostic",))
    if flux["reference_required_weight_coverage_verified"] is not True:
        raise ValueError("current FLUX original reference weights must be actually covered")
    routes = [measured(engine, flux["medians"][engine]["cold_seconds"], flux["medians"][engine]["samples"],
        repeated_prompt_seconds=flux["medians"][engine]["repeated_prompt_seconds"],
        constructor_seconds=flux["medians"][engine]["constructor_seconds"]) for engine in ("native", "mflux")]
    campaigns.append(dict(evidence=binding, model="flux2-klein-9b-current-BF16-native-mflux", geometry=[512, 512],
        steps=4, date="2026-10-11", routes=routes, timing_scope=flux["timing_scope"],
        note="current matched async-build native/mflux BF16 ABBA; complete reference required weights; advantage is1.098x rather than old1.370x; no 9B hybrid validation"))
    flux, binding = source("docs/design/validation/flux9-bf16-cache0-current-20261011.json", ("complete_diagnostic",))
    if flux["reference_cache_mib"] != 0 or flux["reference_required_weight_coverage_verified"] is not True:
        raise ValueError("actual cache0 and complete source coverage required")
    routes = [measured(engine, flux["medians"][engine]["cold_seconds"], flux["medians"][engine]["samples"],
        repeated_prompt_seconds=flux["medians"][engine]["repeated_prompt_seconds"],
        constructor_seconds=flux["medians"][engine]["constructor_seconds"]) for engine in ("native", "mflux")]
    campaigns.append(dict(evidence=binding, model="flux2-klein-9b-cache0-BF16-native-mflux", geometry=[512, 512],
        steps=4, date="2026-10-11", routes=routes, timing_scope=flux["timing_scope"],
        note="same-window cache0 reference candidate; mflux7.351531s beats native7.979642s; retained negative result, not native universal dominance or pooled stock samples"))
    ltx, binding = source("docs/design/validation/ltx-full-configs-current-20261011.json", ("complete_diagnostic",))
    if len(ltx["trials"]) != 16 or len(ltx["quality"]) != 8 or any(row["metrics"]["frame_count_compared"] != 97 for row in ltx["quality"]):
        raise ValueError("complete eight-configuration sixteen-arm full-media audit required")
    for single in (False, True):
        routes = [measured(route["name"], route["cold_seconds"], route["warm_seconds"],
            physical_peaks=route["physical_peaks"], configured=route["configured"], native_timing_boundary=route["native_timing_boundary"])
            for route in ltx["routes"] if route["configured"]["single"] == single]
        campaigns.append(dict(evidence=binding, model="ltx-2.5-full-config-clean-exec" if single else "ltx-2.5-full-config-persistent-batch",
            geometry=[704, 448, 97], steps=11, date="2026-10-11", routes=routes,
            timing_scope="new process conditioning-cache-hit request_wall_estimate; not persistent API wall" if single else "same-process full MP4 request_wall; cold and immediate conditioning-cache-hit warm",
            note="eight source-bound configurations; Sol/text256/INT8 arithmetic separate; two actual warm per config; batch INT8 slower and variable; all32MP4/20memory and E5 raw diagnostics preserved; no physical NE trace or cross-boundary pooling"))
    for path, name in (
        ("docs/design/validation/flux9-compiled-current-20261011.json", "flux2-klein-9b-compiled-GPU-cohort"),
        ("docs/design/validation/flux9-compiled-deferred-current-20261011.json", "flux2-klein-9b-compiled-deferred-encoder-cohort"),
        ("docs/design/validation/flux9-compiled-fused-current-20261011.json", "flux2-klein-9b-compiled-fused-encoder-cohort"),
    ):
        data, binding = source(path, ("complete_diagnostic",))
        if len(data["trials"]) != 6 or any(row.get("actual_compiled_single_blocks") != 96
                for trial in data["trials"] if trial["engine"] == "native-compiled" for row in trial["report"]["runs"]):
            raise ValueError("complete actual9B compiled branch cohort required")
        routes = [measured(engine, data["medians"][engine]["cold_seconds"], data["medians"][engine]["samples"],
            repeated_prompt_seconds=data["medians"][engine]["repeated_prompt_seconds"],
            constructor_seconds=data["medians"][engine]["constructor_seconds"]) for engine in ("native", "native-compiled", "mflux")]
        campaigns.append(dict(evidence=binding, model=name, geometry=[512, 512], steps=4, date="2026-10-11", routes=routes,
            timing_scope=data["timing_scope"], note="same-window eager/actual compiled96 branches/cache0 reference; four fresh per route; parameterized9B geometry, explicit weights; compiled/eager image exact in recorded tasks; reference remains slightly faster, not universal native dominance",
            native_encoder_eval_interval=data.get("native_encoder_eval_interval")))
    grid, binding = source("docs/design/validation/z-private-partition-grid-current-20261011.json", ("complete_diagnostic",))
    if len(grid["trials"]) != 14 or not all(trial["accepted"] for trial in grid["trials"]):
        raise ValueError("complete fourteen-candidate partition grid required")
    for trial in grid["trials"]:
        audit = trial["audit"]
        routes = [measured(row["route"], row["cold_seconds"], row["fresh_warm_seconds"],
            repeated_prompt_seconds=row["repeat_seconds"], physical_peaks=row["physical_peaks"])
            for row in audit["routes"]]
        campaigns.append(dict(evidence=binding, model=f'z-image-turbo-private-a{trial["channels"]}-{trial["join"]}',
            geometry=[512,512], steps=8, date="2026-10-11", routes=routes, timing_scope=grid["shared_source"]["timing_scope"],
            note="same candidate six-process GPU/frozen/Private palindrome; four fresh per route; exact actual channel/join/retry/fallback and original-size quality preserved; candidate winner compared only with its own controls"))
    hybrid, binding = source("docs/design/validation/flux9-hybrid-current-20261012.json", ("complete_diagnostic",))
    if (len(hybrid["trials"]) != 8 or hybrid["order"] != ["native", "native-compiled", "native-hybrid", "mflux", "mflux", "native-hybrid", "native-compiled", "native"] or
            not hybrid["reference_required_weight_coverage_verified"]):
        raise ValueError("complete source-bound9B four-route hybrid palindrome required")
    routes = []
    for engine in ("native", "native-compiled", "native-hybrid", "mflux"):
        declared = hybrid["medians"][engine]
        trials = [trial for trial in hybrid["trials"] if trial["engine"] == engine]
        if any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete9B hybrid cohort memory required")
        if engine == "native-hybrid" and any(row["metrics"]["hybrid"]["runtime_calls_session_total"] != 96 * (row["index"] + 1) or
                row["metrics"]["hybrid"]["runtime_failed"] or not row["metrics"]["hybrid"]["checkpoint_sha256_verified"]
                for trial in trials for row in trial["report"]["runs"]):
            raise ValueError("actual source-verified96 hybrid calls per request required")
        routes.append(measured(engine, declared["cold_seconds"], declared["samples"],
            repeated_prompt_seconds=declared["repeated_prompt_seconds"],
            constructor_seconds=declared["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    campaigns.append(dict(evidence=binding, model="flux2-klein-9b-experimental-hybrid-four-route-cohort",
        geometry=[512,512], steps=4, date="2026-10-12", routes=routes, timing_scope=hybrid["timing_scope"],
        note="eight-process same-window source-bound4096-prefix/8192-GPU complement,96 actual calls/request; INT8 weights/FP16 I/O, not W8A8/native integer MAC; hybrid7.841375/compiled7.843808s differ only0.031%, no stable fresh-request win; cache0 reference7.355288s wins; cold/repeat/pressure and all32PNG/8memory retained, automatic route unchanged"))
    matched,binding = source("docs/design/validation/flux9-matched-dit-q4-current-20261012.json",("complete_diagnostic",))
    if len(matched["trials"]) != 8 or matched["conversion"]["converted_components"] != ["transformer"] or not matched["conversion"]["components"]["transformer"]["full_compressed_payload_equal"]:
        raise ValueError("complete same-payload DiT-only Q4/g32 four-route cohort required")
    routes = []
    for route in ("native-bf16","native-q4","mflux-bf16","mflux-q4"):
        values = matched["medians"][route]
        trials = [trial for trial in matched["trials"] if trial["route"] == route]
        if any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete matched-precision memory proof required")
        routes.append(measured(route,values["cold_seconds"],values["samples"],
            repeated_prompt_seconds=values["repeated_prompt_seconds"],constructor_seconds=values["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    campaigns.append(dict(evidence=binding,model="flux2-klein-9b-matched-dit-q4-g32-cohort",
        geometry=[512,512],steps=4,date="2026-10-12",routes=routes,timing_scope=matched["timing_scope"],
        note="same byte-identical153-linear DiT Q4/g32 planes in native/reference, original BF16 encoder/VAE; all32PNG/8memory complete; nativeQ4 7.986974 versus ownBF16 7.921748s is not speedup, referenceQ4 8.030596 versus BF16 7.379937s is slower; Q4 peaks18.16/19.73GB and compression0, memory-only candidate not full-model quantization or ANE qualification"))
    full,binding = source("docs/design/validation/flux9-matched-full-q4-current-20261012.json",("complete_diagnostic",))
    if (len(full["trials"]) != 8 or full["conversion"]["converted_components"] != ["transformer","text_encoder"] or
            any(full["conversion"]["components"][name]["full_compressed_payload_equal"] is not True for name in ("transformer","text_encoder"))):
        raise ValueError("complete same-payload encoder+DiT Q4/g32 cohort required")
    routes = []
    for route in ("native-bf16","native-q4","mflux-bf16","mflux-q4"):
        values = full["medians"][route]
        trials = [trial for trial in full["trials"] if trial["route"] == route]
        if any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete full-linear matched memory proof required")
        routes.append(measured(route,values["cold_seconds"],values["samples"],
            repeated_prompt_seconds=values["repeated_prompt_seconds"],constructor_seconds=values["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    campaigns.append(dict(evidence=binding,model="flux2-klein-9b-matched-encoder-dit-q4-g32-cohort",
        geometry=[512,512],steps=4,date="2026-10-12",routes=routes,timing_scope=full["timing_scope"],
        note="same153 DiT+252 encoder Q4/g32 linears, BF16 embedding/VAE; complete539+902 saved parameters and all36 encoder layers,32PNG/8memory; nativeQ4 8.005275/ownBF16 7.741839s is3.4% slower, referenceQ4 8.042573/ownBF16 7.346223s slower; nativeQ4 peak11.99GB, lower-memory candidate not fastest, all-module4bit or ANE qualification"))
    decoded,binding = source("docs/design/validation/flux9-predecode-current-20261012.json",("complete_diagnostic",))
    if len(decoded["trials"]) != 10 or "native-predecoded" not in decoded["medians"]:
        raise ValueError("complete actual five-route affine predecode cohort required")
    routes = []
    for route in ("native-bf16","native-q4","native-predecoded","mflux-bf16","mflux-q4"):
        values = decoded["medians"][route]
        trials = [trial for trial in decoded["trials"] if trial["route"] == route]
        if any(trial["memory"]["complete"] is not True for trial in trials):
            raise ValueError("complete predecode cohort process-tree proof required")
        if route == "native-predecoded" and any(row["actual_compiled_single_blocks"] != 96 or
                row["native_event_counts"].get("affine_predecode_projection",0) != (153 if row["index"]==0 else 0)
                for trial in trials for row in trial["report"]["runs"]):
            raise ValueError("actual one-time153 BF16 decodes then96 compiled blocks/request required")
        routes.append(measured(route,values["cold_seconds"],values["samples"],
            repeated_prompt_seconds=values["repeated_prompt_seconds"],constructor_seconds=values["constructor_seconds"],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    campaigns.append(dict(evidence=binding,model="flux2-klein-9b-affine-predecoded-gpu-cohort",
        geometry=[512,512],steps=4,date="2026-10-12",routes=routes,timing_scope=decoded["timing_scope"],
        note="same Q4/g32 stored source,153 DiT projections decoded once to BF16 with additional rounding,96 actual compiled single blocks/request; encoder remainsQ4/BF16 embedding/VAE;40PNG/10memory complete; predecoded7.210639 versus packed8.001289s saves9.882%, peak24.40-24.45GB versus packed11.95-11.97GB; approximate high-memory opt-in, not strictBF16/global win or ANE qualification"))
    return dict(schema="tc-local-base-benchmark-samples-v1", date="2026-10-12", status="partial_expanded_evidence_catalog",
                hardware="Apple M4 Max 64 GiB; dated cohorts retain original build/OS settings",
                qualification_passed=False, goal_complete=False,
                completion_audit="docs/benchmark/COMPLETION_AUDIT.md",
                goal_completion_scope="expanded local family/precision/parameter and best GPU/ANE/framework comparisons remain active; prior-stage completion is not expanded-goal proof; not release qualification",
                expansion_status="docs/benchmark/EXPANSION_STATUS.md",
                campaigns=campaigns,
                missing_current_comparisons=["Qwen further native share/shape and current upstream quantized candidates; reference0/1024/4096MiB and current0.22 BF16/partialFP16 cohorts completed",
                    "Z additional native shape/checkpoint and current upstream quantized candidates; fourteen Private share/join and current0.22 BF16/float16 cohorts completed", "FLUX9B native MLX version/precision/share/shape screening and compatible mlx-serve layouts; current0.22 seven-route BF16/hybrid/Q4/predecode/partialFP16 cohort completed",
                    "LTX best full-media native configurations and runnable reference quantized layouts", "cross-family parameter screening and durable actually-loadable best configurations"],
                unmeasured_or_inapplicable=["current FastH3 INT6 DiT and original H3 video VAE assets absent",
                    "FLUX9B automatic/general-shape hybrid and physical ANE overlap unqualified; exact experimental512/4 prefix measured", "NVFP4 runtime GPU-only", "mlx-serve original FLUX layout rejected and original LTX stock memory preflight rejected",
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
