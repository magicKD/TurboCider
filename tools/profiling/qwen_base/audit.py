#!/usr/bin/env python3
"""Independent CPU artifact/source/observer/image recheck after model windows."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_bf16_framework_screen import write_json
from qwen21_dit_cache_report import image_metrics
from runtime_ane_common import sha256_file
import qwen_base_runtime_screen as base
import collect
import analyze


def pixels(path):
    with Image.open(path) as image:
        image.load()
        if image.size != (512, 512):
            raise ValueError("full512 PNG required")
        return np.asarray(image.convert("RGBA"))


def normalized(path):
    value = json.loads(path.read_text())
    value.pop("output", None)
    return value


def quality(reference, candidate):
    return dict(rgb=image_metrics(reference[:, :, :3], candidate[:, :, :3]),
                alpha_max_abs=int(np.abs(reference[:, :, 3].astype(int) - candidate[:, :, 3]).max()),
                rgba_exact=bool(np.array_equal(reference, candidate)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "native-source-archive", "v1-source-archive", "gpu-baseline", "runtime-baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--failed-surface", type=Path)
    parser.add_argument("--v2-source-archive", type=Path, required=True)
    parser.add_argument("--v3-source-archive", type=Path)
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    with tarfile.open(args.native_source_archive, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        if set(members) != {entry["path"] for entry in manifest["inputs"]["sources"]}:
            raise ValueError("complete native source archive required")
        for entry in manifest["inputs"]["sources"]:
            stream = archive.extractfile(members[entry["path"]])
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != entry["sha256"] or sha256_file(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("native archive/current source differs from linked build")
    historical = {}
    for path in [args.v1_source_archive, args.v2_source_archive] + ([args.v3_source_archive] if args.v3_source_archive else []):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and not Path(member.name).name.startswith("._"):
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())
    library_sha = sha256_file(args.library)

    def window(root, profile):
        summary = json.loads((root / "summary.json").read_text())
        report = json.loads((root / "trial/report.json").read_text())
        if summary["status"] != "complete" or summary["assets"]["library"] != library_sha:
            raise ValueError("complete bound-library process required")
        snapshot = json.loads((root / "source.json").read_text())
        if set(snapshot) != set(summary["source_sha256"]):
            raise ValueError("complete bound tool snapshot required")
        for name, digest in summary["source_sha256"].items():
            raw = snapshot[name].encode()
            if hashlib.sha256(raw).hexdigest() != digest or not (raw == (ROOT / name).read_bytes() or raw in historical.get(name, set())):
                raise ValueError("tool current/historical source changed")
        for name in ("evidence", "report"):
            if sha256_file(root / summary["memory"][name]) != summary["memory"][name + "_sha256"]:
                raise ValueError("memory evidence digest changed")
        if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"] != 0:
            raise ValueError("complete successful owned process window required")
        if profile:
            collect.validate([row["metrics"] for row in report["runs"]], summary["route"], summary["mode"], summary.get("channels", 4096))
            if report["loaded_runtime_id"] != manifest["runtime_build_id"]:
                raise ValueError("actual loaded runtime differs")
            if analyze.native_deltas(report["runs"]) != summary["analysis"]["native_runtime_deltas"]:
                raise ValueError("per-request native counter differences changed")
            if summary["mode"] != "stock":
                observed = json.loads((root / "trial/observer.json").read_text())
                if observed["source_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:
                    raise ValueError("compiled observer source differs")
                actual = analyze.observer_contract(observed, report["runs"], summary["route"] != "gpu")
                previous = copy.deepcopy(summary["analysis"]["observer"])
                for row in previous["requests"]:
                    if "api_host_span_seconds" in row:
                        row["api_or_cb_lifetime_seconds"] = row.pop("api_host_span_seconds")
                if actual != previous:
                    raise ValueError("actual compile/load/model/CB trace recheck differs")
        else:
            base.validate(report, summary["route"], summary["case"])
        images = [pixels(root / f"trial/{i}.png") for i in range(len(report["runs"]))]
        if len(images) == 3 and (not np.array_equal(images[0], images[2]) or np.array_equal(images[0], images[1])):
            raise ValueError("seed change/restoration reused stale output or changed original output")
        hot = [row["request_wall_including_export"] for row in report["runs"][1:]]
        recorded = summary["analysis"].get("hot_median_seconds")
        if hot and statistics.median(hot) != recorded:
            raise ValueError("complete hot request denominator differs")
        return summary, report, images

    gpu = window(args.gpu_baseline, False)
    runtime = window(args.runtime_baseline, False)
    observations, by_config, windows = [], {}, []
    gpu_by_case = {"generate": gpu[2]}
    gpu_root_by_case = {"generate": args.gpu_baseline}
    for root in args.cohort:
        summary, report, images = window(root, True)
        windows.append((root, summary, report, images))
        if summary["route"] == "gpu" and summary["mode"] == "stock":
            gpu_by_case[summary["case"]] = images
            gpu_root_by_case[summary["case"]] = root
        key = summary["case"], summary["route"], summary.get("channels", 4096)
        comparisons = []
        if summary["case"] == "generate":
            if summary["route"] == "cache":
                for i, image in enumerate(images):
                    if normalized(root / f"trial/request-{i}.json") != normalized(args.runtime_baseline / f"trial/request-{i}.json") or not np.array_equal(image, runtime[2][i]):
                        raise ValueError("same runtime arithmetic cache/observer arm changed actual request or RGBA")
            if summary["route"] == "gpu":
                comparisons = [quality(gpu[2][i], image) for i, image in enumerate(images)]
            else:
                comparisons = [quality(gpu[2][i], image) for i, image in enumerate(images)]
        if key in by_config and summary["mode"] in ("stock", "timeline"):
            for first, second in zip(by_config[key], images):
                if not np.array_equal(first, second):
                    raise ValueError("stock/timeline output changed for the same fixed runtime model")
        if summary["mode"] in ("stock", "timeline"):
            by_config[key] = images
        observations.append(dict(cohort=root.name, mode=summary["mode"], route=summary["route"], case=summary["case"], channels=summary.get("channels", 4096),
                                 hot_seconds=[row["request_wall_including_export"] for row in report["runs"][1:]], quality_vs_full_gpu=comparisons,
                                 observer=summary["analysis"].get("observer"), memory=summary["memory"]["verified"],
                                 prefix_snapshot=[dict(enabled="edit prefix KV snapshot " in row["metrics"]["acceleration_selection"],
                                                       hit="edit prefix KV snapshot hit" in row["metrics"]["acceleration_selection"]) for row in report["runs"]],
                                 summary_sha256=sha256_file(root / "summary.json"), load_error=summary.get("load_error")))
    for observation, (root, summary, report, images) in zip(observations, windows):
        reference_images = gpu_by_case.get(summary["case"])
        if reference_images is None:
            continue
        for i in range(len(images)):
            control = normalized(gpu_root_by_case[summary["case"]] / f"trial/request-{i}.json")
            candidate = normalized(root / f"trial/request-{i}.json")
            for request in (control, candidate):
                for key in ("execution", "hybrid_mlp_mode", "ane_manifest"):
                    request.pop(key, None)
            if control != candidate:
                raise ValueError("GPU/runtime comparison changed actual mathematical request inputs")
        observation["quality_vs_full_gpu"] = [quality(reference_images[i], image) for i, image in enumerate(images)]
        hot = observation["hot_seconds"]
        if hot and summary["mode"] == "stock":
            baseline_report = json.loads((gpu_root_by_case[summary["case"]] / "trial/report.json").read_text())
            baseline_median = statistics.median(row["request_wall_including_export"] for row in baseline_report["runs"][1:])
            observation["stock_reduction_vs_same_task_gpu_percent"] = 100 * (1 - statistics.median(hot) / baseline_median)
    failure = None
    if args.failed_surface:
        summary = json.loads((args.failed_surface / "summary.json").read_text())
        rows = json.loads((args.failed_surface / "trial/report.json").read_text())["runs"]
        if summary["status"] != "failed" or len(rows) != 3 or any(row["metrics"]["runtime_backend"] != "mlx_cpp_metal" for row in rows):
            raise ValueError("preserved failed-cache complete GPU fallback evidence required")
        for i, row in enumerate(rows):
            if (row["metrics"]["hybrid"]["runtime_weight"]["failure_reason"] != "W8 graph/banks exceed memory budget"
                    or not np.array_equal(pixels(args.failed_surface / f"trial/{i}.png"), gpu[2][i])):
                raise ValueError("failed cache was incorrectly reclassified as runtime work")
        failure = dict(cohort=args.failed_surface.name, gpu_fallback_outputs=3, runtime_performance_samples=0)
    result = dict(schema="tc-qwen-base-profile-audit-v1", status="complete", qualification_passed=False,
                  library_sha256=library_sha, native_source_hashes_verified=len(manifest["inputs"]["sources"]),
                  native_source_archive_sha256=sha256_file(args.native_source_archive), v1_tool_archive_sha256=sha256_file(args.v1_source_archive),
                  v2_tool_archive_sha256=sha256_file(args.v2_source_archive),
                  v3_tool_archive_sha256=sha256_file(args.v3_source_archive) if args.v3_source_archive else None,
                  gpu_hot_seconds=gpu[0]["analysis"]["full_hot_seconds"], runtime4096_hot_seconds=runtime[0]["analysis"]["full_hot_seconds"],
                  runtime4096_vs_gpu_quality=[quality(gpu[2][i], runtime[2][i]) for i in range(3)],
                  observations=observations, preserved_failed_cache=failure, script_sha256=sha256_file(Path(__file__)),
                  scope="CPU raw-source/first+hot PNG/actual selector-model-CB/native-delta recheck; no new hardware replay, full semantic quality or formal speed qualification")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
