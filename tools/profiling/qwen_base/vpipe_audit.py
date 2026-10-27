#!/usr/bin/env python3
"""Independent CPU recheck of vpipe DiT outputs, routes and diagnostic events."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

import analyze
import vpipe_compare as compare
import vpipe_view
from collect import ROOT, model_snapshot, sha256_file, write_json


def observer_summary(value, samples):
    rows = analyze.observer_evidence(value)
    known = {-3, -2, -1, *range(len(samples))}
    if any(row["request"] not in known for row in rows):
        raise ValueError("unbound reference phase scope")
    hot_models = {row["model"] for row in rows if row["name"] == "ane.evaluate_api" and row["request"] >= 2}
    # This diagnostic specifically covers FFN + QKV long-prefill. Emitter
    # capability models in warmup scope0 are not the two real workload graphs.
    if len(hot_models) != 2:
        raise ValueError("two actually evaluated stable hot runtime models required")
    per_phase = []
    for sample in samples:
        owned = [row for row in rows if row["request"] == sample["index"]]
        calls = Counter(row["name"] for row in owned)
        if not calls["metal.command_buffer"]:
            raise ValueError("reference phase lacks completed Metal observations")
        if sample["phase"] == "decode" and calls["ane.evaluate_api"]:
            raise ValueError("1024 decode unexpectedly evaluated the2048 ANE model")
        if sample["ane_predict_records"] and not calls["ane.evaluate_api"]:
            raise ValueError("claimed ANE arithmetic lacks observed evaluate calls")
        if sample["index"] >= 2 and calls["ane.evaluate_api"] != sample["ane_predict_records"]:
            raise ValueError("hot ANE prediction credits differ from actual evaluate calls")
        if sample["index"] >= 2 and (calls["ane.compile_api"] or calls["ane.load_api"]):
            raise ValueError("reference hot phase compiled or loaded an ANE model")
        per_phase.append(dict(index=sample["index"], phase=sample["phase"], calls=dict(calls),
                              workload_model_ids=sorted({r["model"] for r in owned if r["name"] == "ane.evaluate_api" and r["model"] in hot_models})))
    return dict(hot_runtime_model_ids=sorted(hot_models), hot_no_compile=True, hot_no_load=True,
                phases=per_phase, scope="actual Private client selectors; warm emitter capability tests separate; no ANE hardware duration/placement proof")


def trace(value):
    rows = analyze.observer_evidence(value)
    epoch = min(row["begin_ns"] for row in rows)
    events = []
    for row in rows:
        events.append(dict(name=row["name"], cat="cb-completion-lifetime" if row["name"] == "metal.command_buffer" else "host-api",
                           ph="X", pid=1, tid=row["thread"], ts=(row["begin_ns"] - epoch) / 1000,
                           dur=(row["end_ns"] - row["begin_ns"]) / 1000,
                           args=dict(phase_index=row["request"], model=row["model"], queue=row["queue"])))
        if row["gpu_begin_ns"]:
            events.append(dict(name="GPU CB window", cat="metal-device-window", ph="X", pid=2, tid=row["queue"],
                               ts=(row["gpu_begin_ns"] - epoch) / 1000, dur=(row["gpu_end_ns"] - row["gpu_begin_ns"]) / 1000,
                               args=dict(phase_index=row["request"], scope="CB elapsed window, not shader utilization")))
    return dict(traceEvents=events, displayTimeUnit="ms")


def quality(reference, candidate):
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    return dict(relative_l2=float(np.linalg.norm(delta)) / denominator if denominator else float(np.linalg.norm(delta)),
                max_abs=float(np.max(np.abs(delta))), exact=bool(np.array_equal(reference, candidate)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--source-archive", type=Path, action="append", default=[])
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "references/vpipe")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh derived audit/trace directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    historical = {}
    for path in args.source_archive:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())
    reference = compare.reference_identity(args.reference_root)
    snapshot = model_snapshot(args.model)
    view = vpipe_view.validate(args.model, args.view)
    summaries, reports = {}, {}
    for root in args.cohort:
        summary = json.loads((root / "summary.json").read_text())
        if summary["status"] != "complete" or summary["reference"] != reference or summary["checkpoint_snapshot"] != snapshot:
            raise ValueError("complete unchanged-reference/model process required")
        if summary.get("view_identity") not in (None, view):
            raise ValueError("reference view generation changed")
        sources = json.loads((root / "source.json").read_text())
        if set(sources) != set(summary["sources"]):
            raise ValueError("complete tool snapshots required")
        for name, digest in summary["sources"].items():
            raw = sources[name].encode()
            if hashlib.sha256(raw).hexdigest() != digest or not (raw == (ROOT / name).read_bytes() or raw in historical.get(name, set())):
                raise ValueError("tool current/archive identity differs")
        for name in ("report", "evidence"):
            if sha256_file(root / summary["memory"][name]) != summary["memory"][name + "_sha256"]:
                raise ValueError("memory evidence changed")
        if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"] != 0:
            raise ValueError("complete owned process evidence required")
        report = json.loads((root / "trial/report.json").read_text())
        analysis = compare.validate_report(report, summary["case"], summary["arm"], summary["reps"])
        if analysis != summary["analysis"] or compare.validate_arrays(root / "trial", report) != summary["arrays"]:
            raise ValueError("actual phase/array accounting differs")
        summaries[root], reports[root] = summary, report
    gpu = {s["case"]: root for root, s in summaries.items() if s["arm"] == "gpu" and s["mode"] == "stock"}
    args.output.mkdir(parents=True)
    observations = []
    for root, summary in summaries.items():
        report = reports[root]
        baseline = gpu[summary["case"]]
        pairs = []
        for row in report["samples"]:
            index = row["index"]
            pairs.append(dict(index=index, phase=row["phase"], **quality(
                np.fromfile(baseline / f"trial/sample-{index}.f32", dtype="<f4"),
                np.fromfile(root / f"trial/sample-{index}.f32", dtype="<f4"))))
        observation = dict(cohort=root.name, case=summary["case"], arm=summary["arm"], mode=summary["mode"],
                           phase_analysis=summary["analysis"], quality_vs_gpu=pairs,
                           actual_ane_predict_records=[r["ane_predict_records"] for r in report["samples"]],
                           summary_sha256=sha256_file(root / "summary.json"), memory=summary["memory"]["verified"],
                           load_error=summary.get("load_error"), qualification_passed=False)
        if summary["mode"] == "timeline":
            observer = json.loads((root / "trial/observer.json").read_text())
            if observer["source_sha256"] != summary["sources"]["tools/profiling/qwen_base/observer.mm"]:
                raise ValueError("observer binary source differs")
            observation["observer"] = observer_summary(observer, report["samples"])
            write_json(args.output / f"{root.name}-trace.json", trace(observer))
        observations.append(observation)
    result = dict(schema="tc-vpipe-qwen-component-audit-v1", status="complete", qualification_passed=False,
                  reference_commit=reference["commit"], reference_library_sha256=sorted({s["library_sha256"] for s in summaries.values()}),
                  view_identity=view, observations=observations,
                  source_archive_sha256={str(p): sha256_file(p) for p in args.source_archive},
                  script_sha256=sha256_file(Path(__file__)),
                  scope="DiT component/synthetic inputs only; no full output-image or end-to-end speed claim; adaptive shares alter rounding")
    write_json(args.output / "audit.json", result)
    print(json.dumps(dict(status=result["status"], output=str(args.output), cohorts=len(observations))))


if __name__ == "__main__":
    main()
