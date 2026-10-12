#!/usr/bin/env python3
"""Audit complete derived FLUX4 requests; never reload or retime model weights."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from flux4_derived_hybrid_screen import ORDER, validate_runs
from flux9_bf16_framework_screen import generation, write_json
from flux9_framework_evidence import portable
from runtime_ane_common import sha256_file
from runtime_ane_image_compare import compare_png

ROOT = Path(__file__).resolve().parents[2]


def validate_screen(data):
    if (data.get("status") != "complete_diagnostic" or data.get("order") != list(ORDER) or
            len(data.get("trials", [])) != 4 or data.get("derived_native_comparison_valid") is not True or
            data.get("original_packed_native_retested") is not False or data.get("external_retested") is not False):
        raise ValueError("complete new-derived candidate cohort, no original/native reference retest required")
    for index, trial in enumerate(data["trials"]):
        if (trial["index"], trial["route"]) != (index, ORDER[index]):
            raise ValueError("actual candidate order differs")
        validate_runs(trial["report"]["runs"], trial["route"])
        memory = trial["memory"]["verified"]
        if (memory.get("complete") is not True or memory.get("command_exit_code") != 0 or
                memory.get("allowed_max_gap_ns") != 500_000_000 or memory.get("max_gap_ns", 500_000_001) > 500_000_000 or
                len(trial.get("pngs", [])) != 4):
            raise ValueError("complete independent candidate memory and fourPNG required")
    if [row["route"] for row in data.get("routes", [])] != ["gpu", "hybrid"]:
        raise ValueError("both actual candidate routes required")
    for row in data["routes"]:
        trials = [trial for trial in data["trials"] if trial["route"] == row["route"]]
        samples = [r["request_wall_including_export"] for trial in trials for r in trial["report"]["runs"][1:3]]
        if (len(samples) != 4 or row["fresh_warm_seconds"] != samples or
                row["fresh_warm_median"] != statistics.median(samples) or
                row["cold_seconds"] != [trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials] or
                row["repeat_seconds"] != [trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials]):
            raise ValueError("candidate median must derive from its actual fourfresh samples")


def image_comparison(reference, candidate, kind, request_index):
    result = compare_png(reference, candidate)
    with Image.open(reference) as a, Image.open(candidate) as b:
        left, right = np.array(a), np.array(b)
        result["png_channels"] = left.shape[-1]
        result["png_mode"] = a.mode
        result["metrics"]["rgb_mae_u8"] = float(np.mean(np.abs(left[:, :, :3].astype(float) - right[:, :, :3])))
    return dict(kind=kind, request_index=request_index, reference=str(reference.relative_to(ROOT)),
                candidate=str(candidate.relative_to(ROOT)), **result)


def audit(screen, model, build_manifest):
    data = json.loads((screen / "summary.json").read_text())
    validate_screen(data)
    registration = json.loads((ROOT / "outputs/benchmark-source-registrations/flux4-derived-hybrid.json").read_text())
    for relative, digest in data["source_sha256"].items():
        path = ROOT / relative
        entry = registration.get(str(path))
        if not entry or entry["sha256"] != digest or entry["generation"] != generation(path):
            raise ValueError("first-generation source binding changed; no payload rescans during inference")
    build = json.loads(build_manifest.read_text())
    for row in build["inputs"]["sources"]:
        if sha256_file(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("native build/source mismatch")
    library_path = build_manifest.parent.parent / "libturbocider.dylib"
    if sha256_file(library_path) != data["source_sha256"][str(library_path.relative_to(ROOT))]:
        raise ValueError("actual library differs from source-matched build")
    comparisons = []
    for index, trial in enumerate(data["trials"]):
        output = screen / f"{index}-{trial['route']}"
        if sha256_file(output / "report.json") != trial["report_sha256"]:
            raise ValueError("actual trial report changed")
        for key in ("evidence", "report"):
            if sha256_file(output / trial["memory"][key]) != trial["memory"][key + "_sha256"]:
                raise ValueError("independent memory evidence changed")
        for media in trial["pngs"]:
            path = ROOT / media["path"]
            if sha256_file(path) != media["sha256"]:
                raise ValueError("full generated PNG changed")
            with Image.open(path) as image:
                image.load()
                if image.size != (512, 512):
                    raise ValueError("complete512 PNG required")
    for first, second in ((0, 3), (1, 2), (0, 1), (3, 2)):
        kind = "same_route_forward_reverse" if ORDER[first] == ORDER[second] else "derived_gpu_vs_hybrid"
        for request_index in range(4):
            comparisons.append(image_comparison(screen / f"{first}-{ORDER[first]}" / f"{request_index}.png",
                screen / f"{second}-{ORDER[second]}" / f"{request_index}.png", kind, request_index))
    if any(not row["png_pixel_exact"] for row in comparisons if row["kind"] == "same_route_forward_reverse"):
        raise ValueError("same-route forward/reverse determinism differs")
    previous_path = ROOT / "docs/design/validation/flux4-native-saved-q4-precheck-20261012.json"
    previous = json.loads(previous_path.read_text())
    original_screen = ROOT / previous["evidence"]
    original_screen = original_screen.parent
    for index, route in enumerate(ORDER):
        for request_index in range(4):
            comparisons.append(image_comparison(original_screen / f"{request_index}.png",
                screen / f"{index}-{route}" / f"{request_index}.png", "prior_packed_vs_" + route, request_index))
    source_path = model / "derived-receipt.json"
    source = json.loads(source_path.read_text())
    manifest_paths = [ROOT / path for path in data["source_sha256"] if Path(path).name.startswith("manifest-")]
    if len(manifest_paths) != 1:
        raise ValueError("one actual managed compiled manifest required")
    manifest_path = manifest_paths[0]
    manifest = json.loads(manifest_path.read_text())
    if (manifest["source"]["checkpoint_sha256"] != source["checkpoint_sha256"] or
            source["all_saved_tensor_bitpatterns_verified"] is not True):
        raise ValueError("derived source and compiled partition provenance differ")
    stages = {}
    for route in ("gpu", "hybrid"):
        rows = [row for trial in data["trials"] if trial["route"] == route for row in trial["report"]["runs"][1:3]]
        stages[route] = {stage: statistics.median(row["metrics"]["timings_seconds"][stage] for row in rows)
                        for stage in ("text_encode", "denoise", "vae_decode")}
    return portable(dict(schema="tc-flux4-derived-hybrid-validation-v1", date="2026-10-13",
        status="complete_diagnostic", qualification_passed=False, goal_complete=False,
        derived_native_comparison_valid=True, reference_comparison_valid=False,
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        source_derivation=dict(evidence=str(source_path.relative_to(ROOT)), sha256=sha256_file(source_path), **source),
        build_manifest=str(build_manifest.relative_to(ROOT)), build_manifest_sha256=sha256_file(build_manifest),
        source_inputs_verified=len(build["inputs"]["sources"]), compiled_manifest=str(manifest_path.relative_to(ROOT)),
        source_sha256=data["source_sha256"], order=data["order"], trials=data["trials"], routes=data["routes"],
        comparisons=data["comparisons"], stage_medians_seconds=stages, image_comparisons=comparisons,
        prior_packed_native=dict(evidence=str(previous_path.relative_to(ROOT)), evidence_sha256=sha256_file(previous_path),
            twofresh_seconds=previous["one_process_two_fresh_median_seconds"], retested=False,
            scope="existing packedQ4 precheck reused only; distinct math/lifecycle from new-derived cohort, not a new paired latency baseline"),
        scope=data["timing_scope"], precision_scope=data["precision_scope"], original_packed_native_retested=False,
        external_retested=False, hardware="M4 Max64GiB/macOS26.6.2",
        quality_scope="all32 full-size original-channel PNG comparisons after timing, no resize/alpha dropping; eight same-route pairs exact; numerical similarity/limited prompts, not global/semantic/bit-exact hybrid qualification; runtime quality_validation_enabled=false is not a quality pass"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "model", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    result = audit(args.screen.resolve(strict=True), args.model.resolve(strict=True), args.build_manifest.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(status=result["status"], source_inputs=result["source_inputs_verified"],
        medians={row["route"]: row["fresh_warm_median"] for row in result["routes"]},
        quality={kind:dict(mae_max=max(row["metrics"]["rgb_mae_u8"] for row in result["image_comparisons"] if row["kind"] == kind),
            ssim_min=min(row["metrics"]["rgb_ssim_gaussian11_valid"] for row in result["image_comparisons"] if row["kind"] == kind))
            for kind in {row["kind"] for row in result["image_comparisons"]}})))


if __name__ == "__main__":
    main()
