#!/usr/bin/env python3
"""CPU audit of matched full-base runtime-prefix off/on requests and timeline."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np

import analyze
import collect
from audit import pixels, quality, normalized
from collect import ROOT, model_snapshot, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--reverse-off", type=Path)
    parser.add_argument("--resident", type=Path)
    parser.add_argument("--failed-resident", type=Path, help="preserve the existing public runtime/resident-prefix rejection")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-source-archive", type=Path, required=True)
    parser.add_argument("--tool-source-archive", type=Path, action="append", default=[])
    parser.add_argument("--historical-off", type=Path, help="optional frozen v8 raw image control; not a new timing denominator")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    library_sha = sha256_file(args.library)
    with tarfile.open(args.native_source_archive, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers()}
        if set(members) != {e["path"] for e in manifest["inputs"]["sources"]}:
            raise ValueError("complete native source archive required")
        for entry in manifest["inputs"]["sources"]:
            if (hashlib.sha256(archive.extractfile(members[entry["path"]]).read()).hexdigest() != entry["sha256"] or
                    sha256_file(ROOT / entry["path"]) != entry["sha256"]):
                raise ValueError("native current/archive/source identity differs")
    historical = {}
    for path in args.tool_source_archive:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())

    def window(root, enabled, mode, resident=False):
        summary = json.loads((root / "summary.json").read_text())
        report = json.loads((root / "trial/report.json").read_text())
        if (summary["status"] != "complete" or report["status"] != "complete" or summary["case"] != "edit" or
                summary["mode"] != mode or summary["route"] != "runtime" or summary["channels"] != 6144 or
                summary["runtime_prefix_snapshot"] is not enabled or summary["assets"]["library"] != library_sha or
                summary.get("resident_prefix", False) is not resident or
                report["loaded_runtime_id"] != manifest["runtime_build_id"] or len(report["runs"]) != 3):
            raise ValueError("complete bound-library6144 base edit window required")
        if summary["model_snapshot"] != model_snapshot(ROOT / "models/Comfy-Org-Qwen-Image-2.1"):
            raise ValueError("original model generation changed")
        source = json.loads((root / "source.json").read_text())
        if set(source) != set(summary["source_sha256"]):
            raise ValueError("complete profiling source snapshot required")
        for name, digest in summary["source_sha256"].items():
            raw = source[name].encode()
            if hashlib.sha256(raw).hexdigest() != digest or not (raw == (ROOT / name).read_bytes() or raw in historical.get(name, set())):
                raise ValueError("profiling current/archive source identity differs")
        metrics = [r["metrics"] for r in report["runs"]]
        collect.validate(metrics, "runtime", mode, 6144, enabled, resident)
        if not enabled and not resident:
            previous = 0
            for metric in metrics:
                calls = metric["hybrid"]["runtime_calls_session_total"]
                if metric.get("qwen_runtime_prefix_snapshot") or calls - previous != 1440 or "prefix KV snapshot hit" in metric["acceleration_selection"]:
                    raise ValueError("off control unexpectedly reused prefix state")
                previous = calls
        images = [pixels(root / f"trial/{i}.png") for i in range(3)]
        if np.array_equal(images[0], images[1]) or (not enabled and not resident and not np.array_equal(images[0], images[2])):
            raise ValueError("actual seed change/restoration failed or reused stale output")
        for name in ("report", "evidence"):
            if sha256_file(root / summary["memory"][name]) != summary["memory"][name + "_sha256"]:
                raise ValueError("memory evidence digest differs")
        if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"] != 0:
            raise ValueError("complete successful owned process required")
        hot = [r["request_wall_including_export"] for r in report["runs"][1:]]
        if statistics.median(hot) != summary["analysis"]["hot_median_seconds"]:
            raise ValueError("complete request denominator differs")
        observed = None
        if mode == "timeline":
            value = json.loads((root / "trial/observer.json").read_text())
            if value["source_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:
                raise ValueError("compiled observer source differs")
            observed = analyze.observer_contract(value, report["runs"], True)
            if not observed["hot_no_compile_observed"] or not observed["hot_no_model_load_observed"]:
                raise ValueError("runtime prefix hot requests compiled/loaded models")
            for i, row in enumerate(observed["requests"]):
                if row["api_calls"]["ane.evaluate_api"] != (1443 if i == 0 else 1280):
                    raise ValueError("actual selector evaluations disagree with first+hot native chunks")
        return dict(root=root, summary=summary, report=report, images=images, hot=hot, observer=observed)

    off, on = window(args.off, False, "stock"), window(args.on, True, "stock")
    keys = {k for k in off["summary"]["environment"] if off["summary"]["environment"][k] != on["summary"]["environment"][k]}
    if keys != {"TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT"} or set(off["summary"]["environment"]) != set(on["summary"]["environment"]):
        raise ValueError("only the explicit prefix switch may differ")
    for i in range(3):
        if normalized(args.off / f"trial/request-{i}.json") != normalized(args.on / f"trial/request-{i}.json"):
            raise ValueError("off/on actual requests differ")
    if not np.array_equal(off["images"][0], on["images"][0]):
        raise ValueError("cold miss changed original runtime image arithmetic")
    compared = [quality(off["images"][i], on["images"][i]) for i in range(3)]
    timeline = None
    if args.timeline:
        timed = window(args.timeline, True, "timeline")
        if any(not np.array_equal(a, b) for a, b in zip(timed["images"], on["images"])):
            raise ValueError("observer changed actual same-config complete outputs")
        timeline = dict(cohort=args.timeline.name, observer=timed["observer"], memory=timed["summary"]["memory"]["verified"])
    reverse = None
    if args.reverse_off:
        control = window(args.reverse_off, False, "stock")
        if any(not np.array_equal(a, b) for a, b in zip(control["images"], off["images"])):
            raise ValueError("reverse control changed baseline output")
        reverse = dict(cohort=args.reverse_off.name, hot_seconds=control["hot"], memory=control["summary"]["memory"]["verified"])
    historical_exact = None
    if args.historical_off:
        historical_exact = [bool(np.array_equal(pixels(args.historical_off / f"trial/{i}.png"), off["images"][i])) for i in range(3)]
        if not all(historical_exact):
            raise ValueError("disabled native implementation changed frozen v8 images")
    resident = None
    if args.resident:
        retained = window(args.resident, False, "stock", True)
        for i in range(3):
            if normalized(args.resident / f"trial/request-{i}.json") != normalized(args.off / f"trial/request-{i}.json"):
                raise ValueError("retained-prefix actual request differs")
        resident = dict(cohort=args.resident.name, hot_seconds=retained["hot"],
                        quality_vs_off=[quality(off["images"][i], retained["images"][i]) for i in range(3)],
                        memory=retained["summary"]["memory"]["verified"])
    failed_resident = None
    if args.failed_resident:
        failed = json.loads((args.failed_resident / "summary.json").read_text())
        attempted = json.loads((args.failed_resident / "trial/report.json").read_text())
        if (failed["status"] != "failed" or attempted["status"] != "failed" or attempted["runs"] or
                attempted["loaded_runtime_id"] != manifest["runtime_build_id"] or
                failed["environment"].get("TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV") != "1" or
                "no frozen W8A8 or DBCache/prefix reuse" not in attempted.get("error", "") or
                list((args.failed_resident / "trial").glob("*.png"))):
            raise ValueError("preserved zero-completed-image resident-prefix rejection required")
        failed_resident = dict(cohort=args.failed_resident.name, completed_requests=0, performance_samples=0,
                               reason=attempted["error"], summary_sha256=sha256_file(args.failed_resident / "summary.json"),
                               report_sha256=sha256_file(args.failed_resident / "trial/report.json"))
    result = dict(schema="tc-qwen-runtime-prefix-audit-v1", status="complete", qualification_passed=False,
                  native_sources_verified=len(manifest["inputs"]["sources"]), library_sha256=library_sha,
                  native_source_archive_sha256=sha256_file(args.native_source_archive),
                  tool_source_archive_sha256={str(p): sha256_file(p) for p in args.tool_source_archive},
                  off_hot_seconds=off["hot"], on_hot_seconds=on["hot"],
                  stock_wall_reduction_percent=100 * (1 - statistics.median(on["hot"]) / statistics.median(off["hot"])),
                  incremental_quality_vs_same_runtime_off=compared,
                  cached_vs_original_seed73_rgba_exact=bool(np.array_equal(on["images"][0], on["images"][2])),
                  cold_miss_rgba_exact=True, historical_disabled_v8_rgba_exact=historical_exact,
                  off_memory=off["summary"]["memory"]["verified"], on_memory=on["summary"]["memory"]["verified"],
                  timeline=timeline, reverse_off=reverse, resident_prefix=resident, preserved_failed_resident=failed_resident,
                  source_summaries={str(p): sha256_file(p / "summary.json") for p in (args.off, args.on)},
                  script_sha256=sha256_file(Path(__file__)),
                  scope="matched full40-step standard1024-ref base edit only; CPU recheck, not general semantic quality or exclusive-device qualification")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
