#!/usr/bin/env python3
"""Audit actual Qwen/mflux images and timing; never load/hash model payloads."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from flux9_framework_evidence import portable
from qwen21_mflux_framework_screen import validate_runs
from runtime_ane_common import sha256_file
from runtime_ane_image_compare import rgb_metrics

ROOT = Path(__file__).resolve().parents[2]


def rgba(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in ("RGB", "RGBA") or image.size != (512, 512):
            raise ValueError("complete 512-square 8-bit RGB/RGBA PNG required")
        values = np.array(image)
    if values.dtype != np.uint8:
        raise ValueError("8-bit output required")
    if values.shape == (512, 512, 3):
        values = np.concatenate([values, np.full((512, 512, 1), 255, dtype=np.uint8)], axis=-1)
    if values.shape != (512, 512, 4):
        raise ValueError("full RGB and alpha planes required")
    return values


def compare_rgba(a, b):
    def white(value):
        alpha = value[:, :, 3:4].astype(np.float64) / 255
        return np.round(value[:, :, :3].astype(np.float64) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    alpha_a, alpha_b = a[:, :, 3], b[:, :, 3]
    return dict(raw_rgb_metrics=rgb_metrics(a[:, :, :3], b[:, :, :3]),
                white_composited_rgb_metrics=rgb_metrics(white(a), white(b)),
                raw_rgb_mae_255=float(np.mean(np.abs(a[:, :, :3].astype(np.float64) - b[:, :, :3].astype(np.float64)))),
                rgba_pixel_exact=bool(np.array_equal(a, b)),
                alpha=dict(reference_min=int(alpha_a.min()), reference_max=int(alpha_a.max()),
                    candidate_min=int(alpha_b.min()), candidate_max=int(alpha_b.max()),
                    reference_nonopaque_pixels=int(np.count_nonzero(alpha_a != 255)),
                    candidate_nonopaque_pixels=int(np.count_nonzero(alpha_b != 255)),
                    mae_255=float(np.mean(np.abs(alpha_a.astype(np.float64) - alpha_b.astype(np.float64))))))


def audit(screen, layout, build):
    data = json.loads((screen / "summary.json").read_text())
    if data["status"] not in ("complete_precheck_only", "complete_diagnostic") or len(data["trials"]) != len(data["order"]):
        raise ValueError("terminal complete image experiment required")
    if data["status"] == "complete_diagnostic" and (data["steps"], data["count"]) != (40, 4):
        raise ValueError("short prechecks cannot be full benchmark evidence")
    manifest = json.loads(build.read_text())
    if any(sha256_file(ROOT / row["path"]) != row["sha256"] for row in manifest["inputs"]["sources"]):
        raise ValueError("native source/build mismatch")
    receipt = json.loads((layout / "layout-receipt.json").read_text())
    if sha256_file(layout / "layout-receipt.json") != data["layout_receipt_sha256"] or receipt["status"] != "complete_layout_only":
        raise ValueError("same verified layout conversion required")
    result = dict(schema="tc-qwen21-mflux-framework-validation-v1", date="2026-10-11", status=data["status"],
                  qualification_passed=False, goal_complete=False, hardware="M4 Max64GiB/macOS26.6.2",
                  steps=data["steps"], requests_per_process=data["count"], order=data["order"],
                  evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
                  native_inputs_matched=len(manifest["inputs"]["sources"]), library_sha256=data["library_sha256"],
                  model_sha256=data["model_sha256"], mflux_commit=data["mflux_commit"], mflux_source_snapshot=data["mflux_source_snapshot"],
                  layout_proof=dict(receipt_sha256=data["layout_receipt_sha256"], components=[dict(kind=item["kind"],
                      source_sha256=item["source_sha256"], derived_sha256=item["derived_sha256"], payload_sha256=item["payload_sha256"],
                      full_payload_equal=item["full_payload_equal"], tensor_bindings=len(item["tensor_bindings"]),
                      preparation_seconds=item["preparation_seconds"]) for item in receipt["files"]]),
                  timing_scope=data["timing_scope"], representation_scope=data["representation_scope"],
                  source_policy=data["source_policy"], trials=[], routes=[], quality=[])
    for trial in data["trials"]:
        validate_runs(trial["report"], trial["route"], data["steps"], data["count"])
        directory = screen / f'{trial["index"]}-{trial["route"]}'
        memory = trial["memory"]
        if memory["verified"]["complete"] is not True or memory["verified"]["command_exit_code"] != 0:
            raise ValueError("independent complete process-tree proof required")
        for name in ("evidence", "report"):
            if sha256_file(directory / memory[name]) != memory[name + "_sha256"]:
                raise ValueError("raw memory evidence changed")
        for index, png in enumerate(trial["pngs"]):
            path = directory / f"{index}.png"
            if sha256_file(path) != png["file_sha256"]:
                raise ValueError("PNG output changed")
            rgba(path)
        result["trials"].append(dict(route=trial["route"], index=trial["index"], report=portable(trial["report"]),
            memory={name: memory["verified"][name] for name in ("complete", "tree_peak_phys_footprint_bytes", "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")},
            memory_evidence_sha256=memory["evidence_sha256"], memory_report_sha256=memory["report_sha256"], pngs=trial["pngs"]))
    if data["count"] == 4:
        for route in sorted(set(data["order"])):
            trials = [trial for trial in result["trials"] if trial["route"] == route]
            samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
            result["routes"].append(dict(route=route, cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                constructor_seconds=[trial["report"]["constructor_seconds"] for trial in trials], fresh_warm_seconds=samples,
                fresh_warm_median=statistics.median(samples), repeat_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials]))
    native = next((trial for trial in result["trials"] if trial["route"] == "native-gpu"), None)
    if native:
        directory = screen / f'{native["index"]}-native-gpu'
        for trial in result["trials"]:
            if trial["route"] == "native-gpu":
                continue
            candidate = screen / f'{trial["index"]}-{trial["route"]}'
            for index in range(data["count"]):
                a, b = rgba(directory / f"{index}.png"), rgba(candidate / f"{index}.png")
                result["quality"].append(dict(route=trial["route"], process_index=trial["index"], request_index=index,
                    comparison=compare_rgba(a, b)))
    result["quality_scope"] = "native PNG retains generated alpha; reference RGB has implicit opaque alpha; raw RGB, alpha and white-composited RGB separately reported; same-size outputs, no resize; cross-engine seed does not prove identical noise or full output-feature equivalence"
    if "/Users/" in json.dumps(result):
        raise ValueError("portable evidence contains machine-specific paths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "layout", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh portable evidence output required")
    result = audit(args.screen.resolve(), args.layout.resolve(), args.build_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(status=result["status"], routes=result["routes"])))


if __name__ == "__main__":
    main()
