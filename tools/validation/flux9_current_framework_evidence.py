#!/usr/bin/env python3
"""Independently audit full current FLUX recipes; never load/hash model payloads."""
import argparse
from datetime import date
import json
from pathlib import Path
import statistics

import numpy as np

from flux9_current_framework_screen import ORDER, QUANTIZED, validate_report
from flux9_framework_evidence import rgb
from flux9_matched_affine_evidence import portable_tree
from runtime_ane_common import sha256_file
from runtime_ane_image_compare import rgb_metrics

ROOT = Path(__file__).resolve().parents[2]
MEMORY_KEYS = ("complete", "tree_peak_phys_footprint_bytes", "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")


def cohort(data):
    if (data.get("status") != "complete_diagnostic" or data.get("count") != 4 or
            data.get("order") != list(ORDER) or len(data.get("trials", [])) != len(ORDER) or
            data.get("mflux_package_version") != "0.22.0" or data.get("reference_cache_mib") != 0 or
            data.get("conversion", {}).get("converted_components") != ["transformer", "text_encoder"]):
        raise ValueError("complete exact current seven-route source-bound palindrome required")
    conversion = data["conversion"]
    if conversion.get("status") != "complete_conversion_only" or any(
            conversion["components"][name].get("full_compressed_payload_equal") is not True
            for name in ("transformer", "text_encoder")):
        raise ValueError("whole matched original-Q4 payload equality required")
    for index, trial in enumerate(data["trials"]):
        if trial["index"] != index or trial["route"] != ORDER[index]:
            raise ValueError("actual full measured route order differs")
        validate_report(trial["report"], trial["route"])
        expected = conversion["converted_components"] if trial["route"] in QUANTIZED else []
        if trial["report"].get("quantized_components") != expected:
            raise ValueError("actual converted component scope differs")
    for route in dict.fromkeys(ORDER):
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
        declared = data["medians"][route]
        if declared["samples"] != samples or declared["fresh_warm_seconds"] != statistics.median(samples):
            raise ValueError("all four actual fresh samples must determine each median")
    return data["medians"]


def audit(screen, build_manifest, evidence_date):
    date.fromisoformat(evidence_date)
    data = json.loads((screen / "summary.json").read_text())
    medians = cohort(data)
    build = json.loads(build_manifest.read_text())
    if any(sha256_file(ROOT / row["path"]) != row["sha256"] for row in build["inputs"]["sources"]):
        raise ValueError("native build/source identity differs")
    library = build_manifest.parent.parent / "libturbocider.dylib"
    if sha256_file(library) != data["local_source_sha256"].get(str(library.relative_to(ROOT))):
        raise ValueError("actual native library differs")
    for name, digest in data["local_source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("actual comparison helper source changed")
    for name, digest in data["ane_source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("retained compiled CoreML artifacts changed")
    manifest = Path(data["ane_manifest"])
    if data["ane_source_sha256"].get(str(manifest.relative_to(ROOT))) != sha256_file(manifest):
        raise ValueError("bound actual hybrid manifest required")
    metadata = json.loads(manifest.read_text())
    source = metadata["source"]
    index = ROOT / "models/FLUX.2-klein-9B/transformer/diffusion_pytorch_model.safetensors.index.json"
    if (source.get("blocks") != list(range(24)) or source.get("checkpoint") != str(index) or
            source.get("checkpoint_sha256") != data["model_sha256"].get(str(index.relative_to(ROOT))) or
            len(source.get("checkpoint_shards", {})) != 2):
        raise ValueError("same original complete9B checkpoint-bound hybrid source required")
    for name, identity in source["checkpoint_shards"].items():
        if identity["sha256"] != data["model_sha256"].get(str((index.parent / name).relative_to(ROOT))):
            raise ValueError("complete original shard provenance differs")
    result = dict(schema="tc-flux9-current-framework-validation-v1", status="complete_diagnostic", date=evidence_date,
        goal_complete=False, qualification_passed=False, hardware="M4 Max64GiB/macOS26.6.2",
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        task=data["task"], order=data["order"], count=4, medians=medians,
        mflux_package_version=data["mflux_package_version"], mflux_commit=data["mflux_commit"],
        mflux_source_sha256=data["mflux_source_sha256"], native_inputs_matched=len(build["inputs"]["sources"]),
        local_source_sha256=data["local_source_sha256"], model_sha256=data["model_sha256"],
        ane_manifest=data["ane_manifest"], ane_source_sha256=data["ane_source_sha256"],
        conversion=data["conversion"], reference_cache_mib=0, timing_scope=data["timing_scope"],
        source_policy=data["source_policy"], representation_scope=data["representation_scope"],
        reference_comparison_valid=True, trials=[], quality=[], output_checks=[])
    images = {}
    for trial in data["trials"]:
        directory = screen / f'{trial["index"]}-{trial["route"]}'
        if json.loads((directory / "report.json").read_text()) != trial["report"]:
            raise ValueError("raw actual child report changed")
        memory = trial["memory"]
        verified = memory["verified"]
        if verified["complete"] is not True or verified["command_exit_code"] != 0 or verified["reasons"]:
            raise ValueError("complete successful independent process-tree memory required")
        for name in ("report", "evidence"):
            if sha256_file(directory / memory[name]) != memory[name + "_sha256"]:
                raise ValueError("raw memory evidence changed")
        if len(trial["png_sha256"]) != 4:
            raise ValueError("all four original-size PNGs required")
        for request_index, digest in enumerate(trial["png_sha256"]):
            path = directory / f"{request_index}.png"
            if sha256_file(path) != digest:
                raise ValueError("actual full PNG changed")
            images[trial["index"], request_index] = rgb(path)
            request = json.loads((directory / f"request-{request_index}.json").read_text())
            if (request["model"], request["operation"], request["width"], request["height"], request["steps"], request["seed"]) != (
                    "flux2-klein-9b", "image.generate", 512, 512, 4, 42) or request.get("loras"):
                raise ValueError("actual complete base-only identical task required")
            if trial["route"] in ("native-q4", "native-predecoded") and (
                    request.get("allow_approximation") is not True or
                    request.get("compile_gpu") is not (trial["route"] == "native-predecoded")):
                raise ValueError("actual packed/predecoded approximate GPU recipe differs")
            if trial["route"] == "native-hybrid" and (
                    request.get("execution") != "gpu_ane" or request.get("ane_manifest") != str(manifest) or
                    trial["report"].get("ane_manifest_sha256") != sha256_file(manifest)):
                raise ValueError("actual successful source-bound hybrid request differs")
        result["trials"].append(dict(index=trial["index"], route=trial["route"], report=trial["report"],
            memory={name: verified[name] for name in MEMORY_KEYS},
            memory_evidence_sha256=memory["evidence_sha256"], memory_report_sha256=memory["report_sha256"],
            png_sha256=trial["png_sha256"]))
    positions = {route: [trial["index"] for trial in data["trials"] if trial["route"] == route] for route in dict.fromkeys(ORDER)}
    pairs = (("native-bf16", "native-hybrid", "native-hybrid-vs-bf16"),
             ("native-bf16", "native-q4", "native-q4-vs-bf16"),
             ("native-q4", "native-predecoded", "predecoded-vs-packed-q4"),
             ("native-bf16", "native-predecoded", "predecoded-vs-original-bf16"),
             ("mflux-bf16", "mflux-fp16", "reference-partial-fp16-vs-bf16"),
             ("mflux-bf16", "mflux-q4", "reference-q4-vs-bf16"),
             ("native-bf16", "mflux-bf16", "native-vs-reference-bf16"),
             ("native-q4", "mflux-q4", "native-vs-reference-matched-q4"))
    for left, right, label in pairs:
        for arm in range(2):
            a, b = positions[left][arm], positions[right][arm]
            for request_index in range(4):
                x, y = images[a, request_index], images[b, request_index]
                result["quality"].append(dict(comparison=label, left_process=a, right_process=b, request_index=request_index,
                    mae_u8=float(np.mean(np.abs(x.astype(np.float64) - y))), **rgb_metrics(x, y)))
    for route, (left, right) in positions.items():
        result["output_checks"].append(dict(route=route, exact_requests=[
            bool(np.array_equal(images[left, index], images[right, index])) for index in range(4)]))
    values = {route: item["fresh_warm_seconds"] for route, item in medians.items()}
    result["comparisons"] = dict(reference_bf16_over_native_bf16=values["mflux-bf16"] / values["native-bf16"],
        reference_fp16_over_native_predecoded=values["mflux-fp16"] / values["native-predecoded"],
        reference_q4_over_native_q4=values["mflux-q4"] / values["native-q4"],
        reference_fp16_reduction_vs_bf16_percent=100 * (1 - values["mflux-fp16"] / values["mflux-bf16"]),
        native_hybrid_reduction_vs_compiled_percent=100 * (1 - values["native-hybrid"] / values["native-bf16"]),
        native_predecode_reduction_vs_packed_percent=100 * (1 - values["native-predecoded"] / values["native-q4"]))
    result["quality_scope"] = "all original-size RGB/native opaque alpha; same-engine precision drift separate from cross-engine noise/arithmetic; not strictBF16, broad perceptual, native integer-MAC or physical-ANE-overlap qualification"
    result = portable_tree(result)
    if "/Users/" in json.dumps(result):
        raise ValueError("portable evidence contains machine-specific paths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh portable evidence output required")
    result = audit(args.screen.resolve(), args.build_manifest.resolve(), args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(status=result["status"], comparisons=result["comparisons"])))


if __name__ == "__main__":
    main()
