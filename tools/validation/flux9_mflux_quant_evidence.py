#!/usr/bin/env python3
"""Publish independently checked stock FLUX precision samples, not native ranks."""
import argparse
import json
import math
from pathlib import Path
import statistics

from flux9_framework_evidence import portable
from flux9_mflux_quant_screen import BITS, validate_report
from qwen21_mflux_framework_evidence import compare_rgba, rgba
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
MEMORY_KEYS = ("complete", "tree_peak_phys_footprint_bytes", "tree_peak_rss_bytes",
               "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")


def cohort(data):
    expected = ["bf16", "q4", "q8", "q8", "q4", "bf16"]
    if data.get("status") != "complete_diagnostic" or data.get("order") != expected or len(data.get("trials", [])) != 6:
        raise ValueError("complete exact stock three-precision palindrome required")
    routes = []
    for index, trial in enumerate(data["trials"]):
        if trial["index"] != index or trial["mode"] != expected[index]:
            raise ValueError("actual precision order differs")
        report = trial["report"]
        validate_report(report, trial["mode"])
        coverage = report.get("required_weight_coverage")
        if not isinstance(coverage, dict) or set(coverage) != {"vae", "transformer", "text_encoder"} or any(
                type(row.get("required_targets")) is not int or row["required_targets"] <= 0 or
                row.get("covered_targets") != row["required_targets"] or row.get("missing") != [] for row in coverage.values()):
            raise ValueError("complete actual source parameter coverage required")
        for row in report["runs"]:
            value = row["request_wall_including_export"]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive full PNG request wall required")
    for mode in BITS:
        trials = [trial for trial in data["trials"] if trial["mode"] == mode]
        samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
        declared = next(row for row in data["routes"] if row["mode"] == mode)
        if declared["fresh_warm_seconds"] != samples or declared["fresh_warm_median"] != statistics.median(samples):
            raise ValueError("all actual fresh samples must match the declared median")
        routes.append(dict(route="mflux-" + mode, fresh_warm_seconds=samples, fresh_warm_median=statistics.median(samples),
            cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
            repeat_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
            constructor_seconds=[trial["report"]["constructor_seconds"] for trial in trials],
            physical_peaks=[trial["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    return routes


def audit(screen):
    data = json.loads((screen / "summary.json").read_text())
    routes = cohort(data)
    source_hashes = {portable(path): digest for path, digest in data["source_sha256"].items()}
    result = dict(schema="tc-flux9-stock-precision-cohort-v1", status="complete_diagnostic", date="2026-10-11",
        qualification_passed=False, goal_complete=False, native_latency_ranking_valid=False,
        hardware="M4 Max64GiB/macOS26.6.2", steps=4, requests_per_process=4, geometry=[512, 512], seed=42,
        order=data["order"], routes=routes, source_sha256=source_hashes,
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        scope=data["scope"], timing_scope="stock mflux full PNG wall; constructor separately; cold/two fresh/repeat per process",
        source_policy="initial campaign source SHA; metadata guard before and after each process; no per-inference model SHA",
        representation_scope="same BF16 source; eligible mflux text/DiT modules affine quantized; not same arithmetic as native BF16/GGUF",
        trials=[], quality=[])
    images = {}
    for trial in data["trials"]:
        directory = screen / f'{trial["index"]}-{trial["mode"]}'
        memory = trial["memory"]
        checked = memory["verified"]
        if checked["complete"] is not True or checked["command_exit_code"] != 0 or checked["reasons"]:
            raise ValueError("complete successful independently verified memory required")
        for name in ("evidence", "report"):
            if sha256_file(directory / memory[name]) != memory[name + "_sha256"]:
                raise ValueError("raw memory evidence changed")
        if len(trial["png_sha256"]) != 4:
            raise ValueError("four actual full PNG receipts required")
        for index, digest in enumerate(trial["png_sha256"]):
            path = directory / f"{index}.png"
            if sha256_file(path) != digest:
                raise ValueError("PNG output changed")
            images[trial["index"], index] = rgba(path)
            request = json.loads((directory / f"request-{index}.json").read_text())
            if (request["model"], request["operation"], request["width"], request["height"], request["steps"], request["seed"], request["quantize"]) != (
                    "flux2-klein-9b", "image.generate", 512, 512, 4, 42, BITS[trial["mode"]]):
                raise ValueError("actual requested source task differs")
        result["trials"].append(dict(index=trial["index"], mode=trial["mode"], report=portable(trial["report"]),
            memory={key: checked[key] for key in MEMORY_KEYS}, memory_evidence_sha256=memory["evidence_sha256"],
            memory_report_sha256=memory["report_sha256"], png_sha256=trial["png_sha256"]))
    for trial in data["trials"][1:]:
        for index in range(4):
            result["quality"].append(dict(mode=trial["mode"], process_index=trial["index"], request_index=index,
                comparison=compare_rgba(images[0, index], images[trial["index"], index])))
    result["quality_scope"] = "same stock engine/source, original-size RGBA/RGB with alpha retained; quantized representation drift diagnostic, not native parity or broad perceptual qualification"
    if "/Users/" in json.dumps(result):
        raise ValueError("machine-specific path in portable evidence")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh evidence output required")
    result = audit(args.screen.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(status=result["status"], routes=result["routes"])))


if __name__ == "__main__":
    main()
