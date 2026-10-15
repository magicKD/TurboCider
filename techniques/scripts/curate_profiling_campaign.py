#!/usr/bin/env python3
"""Verify selected local profiling receipts; export a portable evidence subset.

Reads only explicit campaign inputs. Does not build, execute, replay or search
source directories. Captures and compiler caches remain ignored and local.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from normalize_gputrace import AUDITED_COMMIT, normalize, reject_constant


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def relative_file(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping selected evidence asset")
    return path


def read(path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("JSON exceeds the evidence adapter cap")
    return json.loads(path.read_bytes(), parse_constant=reject_constant)


def receipt(folder):
    path = folder / "receipt.json"
    value = read(path)
    return {"status": value["status"], "returncode": value.get("returncode"),
            "owned_process_exit_confirmed": value.get("owned_process_exit_confirmed"), "sha256": digest(path)}


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def native_summary(document):
    if document.get("schema") != "turbocider-metal-teaching-lab-v1":
        raise ValueError("native profiling schema required")
    if not all(document.get(key) is True for key in
               ("actual_metal_dispatch", "all_outputs_finite", "tail_guards_intact", "source_inputs_unchanged")):
        raise ValueError("native output, guards and source identity must be validated")
    rows = document["samples"]
    warmups = document["warmup_samples"]
    repeats = document["arguments"]["repeats"]
    if type(repeats) is not int or not 4 <= repeats <= 64 or repeats % 4:
        raise ValueError("bounded ABBA native repeats required")
    if [row["trial"] for row in rows] != list(range(document["arguments"]["repeats"])) or [row["trial"] for row in warmups] != [-4, -3, -2, -1]:
        raise ValueError("missing, duplicate or reordered native samples")
    pairs = {}
    mode = document.get("diagnostic_mode")
    if mode not in ("none", "counter-stage", "counter-dispatch", "capture"):
        raise ValueError("unknown native diagnostic mode")
    for row in warmups + rows:
        if not positive(row["wall_seconds"]) or not positive(row["gpu_command_seconds_sum"]):
            raise ValueError("positive finite measured host and completed-command durations required")
        if row in rows:
            if (row["pair"] != row["trial"] // 2 or row["warmup"] is not False
                    or row["variant"] != ("candidate" if row["trial"] % 4 in (1, 2) else "baseline")):
                raise ValueError("native trial order/pair metadata changed")
            pair = pairs.setdefault(row["pair"], {})
            if row["variant"] in pair:
                raise ValueError("duplicate paired arm")
            pair[row["variant"]] = row
        counter = row.get("counter_diagnostic")
        if mode.startswith("counter-") and (not counter or counter.get("sampling_point") !=
                ("stage-boundary" if mode == "counter-stage" else "dispatch-boundary-barrier")):
            raise ValueError("counter diagnostic is missing or has a different sampling point")
        if mode in ("none", "capture") and counter is not None:
            raise ValueError("unexpected instrumentation in the declared native mode")
        if counter:
            for sample in counter["samples"]:
                start, end = sample["start_raw"], sample["end_raw"]
                valid = (type(start) is int and type(end) is int and 0 < start < end < 2**64 - 1)
                if sample["valid"] is not valid or sample["delta_raw"] != (end - start if valid else None):
                    raise ValueError("raw counter validity/delta marker disagrees with samples")
            if counter["duration_seconds"] is not None:
                raise ValueError("raw counter clock unit has not been qualified in this campaign")
    if any(set(pair) != {"baseline", "candidate"} or pair["baseline"]["input_generation"] != pair["candidate"]["input_generation"] for pair in pairs.values()):
        raise ValueError("unmatched native A/B inputs")
    formal = document.get("diagnostic_mode") == "none" and document.get("formal_performance_score") is True
    result = {"variant_median_host_ms": {
        arm: statistics.median(row["wall_seconds"] for row in rows if row["variant"] == arm) * 1000
        for arm in ("baseline", "candidate")}, "formal_operator_score": formal,
        "source_kind": "derived", "production_qualified": False}
    if formal:
        ratios = [pair["baseline"]["wall_seconds"] / pair["candidate"]["wall_seconds"] for pair in pairs.values()]
        result["paired_host_ratios"] = ratios
        result["median_paired_host_ratio"] = statistics.median(ratios)
    else:
        result["median_paired_host_ratio"] = None
    return result


def curate(root, native_ids, captures):
    root = root.resolve()
    parser_root = root / "gputrace-offline-v1"
    if len(native_ids) != len(set(native_ids)) or len(captures) != len(set(captures)):
        raise ValueError("duplicate selected run")
    names = [*native_ids, *(name for selection in captures for name in selection.split(":"))]
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", name) for name in names):
        raise ValueError("selected run IDs must be bounded local names, not paths")
    result = {"schema": "tc-gpu-profiling-evidence-v1", "date": "2026-10-16",
              "production_qualified": False, "curator_sha256": digest(Path(__file__)),
              "normalizer_sha256": digest(Path(__file__).with_name("normalize_gputrace.py")),
              "parser": {"commit": AUDITED_COMMIT, "binary_sha256": digest(parser_root / "gputrace"),
                         "build": receipt(parser_root / "build"), "selected_parser_tests": receipt(parser_root / "parser-tests"),
                         "test_scope": "internal/timing and internal/trace only; nonverbose logs do not establish skip counts",
                         "build_info_sha256": digest(parser_root / "build-info/stdout.log"),
                         "go_mod_sha256": digest(parser_root / "source/go.mod"), "go_sum_sha256": digest(parser_root / "source/go.sum")},
              "instruments_probe": receipt(root / "xctrace-capability-v1"),
              "native_runs": [], "captures": []}
    for identity in native_ids:
        folder = relative_file(root, identity + "/report.json").parent
        document = read(folder / "report.json")
        record = {"id": identity, "report_sha256": digest(folder / "report.json"), "launch": receipt(folder)}
        for snapshot in document.get("source_snapshots", []):
            if digest(relative_file(folder, snapshot["path"])) != snapshot["sha256"]:
                raise ValueError("native source snapshot no longer matches its report")
        keep = ("case", "device", "os", "shape", "vector_count", "threads", "arguments", "diagnostic_mode",
                "formal_performance_score", "captured_trial", "source_sha256", "source_snapshots", "source_inputs_unchanged",
                "binary_sha256", "counter_capabilities", "samples", "warmup_samples", "maximum_abs_error",
                "all_outputs_finite", "tail_guards_intact", "actual_metal_dispatch", "source_inputs_synthetic", "scope")
        record.update({key: document[key] for key in keep if key in document})
        if record["launch"]["returncode"] == 0:
            record["summary"] = native_summary(document)
        else:
            record.update(status=document.get("status"), summary=None,
                          error_class="requested-counter-point-unavailable" if "sampling point unsupported" in document.get("error", "") else "unclassified-native-failure")
        result["native_runs"].append(record)
    for selection in captures:
        identity, parse_key = selection.split(":")
        folder = relative_file(root, identity + "/report.json").parent
        document = read(folder / "report.json")
        if "runner_sha256" in document and digest(folder / "mlx_capture_probe.py") != document["runner_sha256"]:
            raise ValueError("capture runner snapshot no longer matches its report")
        inventory = document.get("capture_files", document.get("capture_artifacts", []))
        if not inventory or not any(type(asset.get("bytes")) is int and asset["bytes"] > 0 for asset in inventory):
            raise ValueError("capture artifact inventory is empty")
        for asset in inventory:
            path = relative_file(folder, asset["path"])
            if digest(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                raise ValueError("capture content changed after collection")
        inventory_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
        stats_folder = parser_root / ("stats-" + parse_key)
        timing_folder = parser_root / ("timing-" + parse_key)
        stats_receipt, timing_receipt = receipt(stats_folder), receipt(timing_folder)
        record = {"id": identity, "report_sha256": digest(folder / "report.json"),
                  "controls": document.get("controls"), "source_runner_sha256": document.get("runner_sha256"),
                  "max_abs_vs_fp64": document.get("max_abs_vs_fp64"), "unprofiled_host_samples": document.get("samples") if identity.startswith("mlx-") else None,
                  "capture_inventory_sha256": inventory_hash, "capture_assets_verified": len(inventory),
                  "reported_logical_bytes_including_aliases": sum(asset["bytes"] for asset in inventory),
                  "raw_capture_metadata_exported": False, "stats_launch": stats_receipt, "timing_launch": timing_receipt}
        if stats_receipt["returncode"] == 0:
            stats = read(stats_folder / "stdout.log")["statistics"]
            record["parsed_structure"] = {key: stats.get(key) for key in
                                           ("command_buffers", "compute_encoders", "dispatch_calls", "unique_kernels", "total_records")}
        else:
            record["parsed_structure"] = None
            record["stats_error_class"] = "capture-header-too-small" if "capture data too small for header" in (stats_folder / "stderr.log").read_text() else "unclassified-parser-failure"
        if timing_receipt["returncode"] != 0:
            raise ValueError("retain and classify failed timing extraction before curating")
        timing_path = timing_folder / "stdout.log"
        normalized = normalize(read(timing_path), AUDITED_COMMIT)
        record["raw_timing_sha256"] = digest(timing_path)
        record["normalized_timing"] = {key: value for key, value in normalized.items() if key != "events"}
        record["normalized_timing"].update(event_prefix=normalized["events"][:8],
                                          event_rows_not_embedded=max(0, len(normalized["events"]) - 8))
        record["normalized_timing_content_sha256"] = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
        result["captures"].append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native", action="append", default=[])
    parser.add_argument("--capture", action="append", default=[], help="run-id:parser-run-key")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = curate(args.root, args.native, args.capture)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print("verified selected profiling evidence; raw captures remain local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
