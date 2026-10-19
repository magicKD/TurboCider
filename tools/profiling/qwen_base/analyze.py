#!/usr/bin/env python3
"""CPU-only phase/compile/queue timeline analysis. No sums of overlapping spans."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def union_ns(intervals):
    intervals = sorted(intervals)
    total, end = 0, None
    for begin, finish in intervals:
        if not 0 <= begin <= finish:
            raise ValueError("invalid interval")
        total += finish - max(begin, end if end is not None else begin) if end is None or finish > end else 0
        end = finish if end is None else max(end, finish)
    return total


def observer_evidence(value):
    """Common raw-record gates, without assuming a particular model route."""
    if (value.get("enabled") is not True or value.get("drained") is not True or value.get("restored_imps") is not True
            or value.get("pending") != 0 or value.get("dropped") != 0):
        raise ValueError("complete enabled non-dropped observer evidence required")
    rows = value.get("events", [])
    counts = Counter(row["name"] for row in rows)
    if dict(counts) != value["counts"]:
        raise ValueError("actual selector/CB event counters differ")
    calibration = [row for row in rows if row["request"] == -2 and row["name"] == "metal.command_buffer"]
    if len(calibration) != 1 or not calibration[0]["gpu_begin_ns"]:
        raise ValueError("one real Metal clock/completion calibration required")
    for row in rows:
        if (row.get("ok") is not True or type(row["begin_ns"]) is not int or type(row["end_ns"]) is not int
                or not 0 < row["begin_ns"] <= row["end_ns"]):
            raise ValueError("failed/incomplete host API/CB record")
        if row["name"] == "metal.command_buffer" and row["gpu_begin_ns"]:
            if not row["begin_ns"] - 2000 <= row["gpu_begin_ns"] <= row["gpu_end_ns"] <= row["end_ns"] + 2000:
                raise ValueError("Metal GPU/host clock domains or completion window disagree")
    return rows


def observer_contract(value, runs, runtime, *, expected_models=1):
    rows = observer_evidence(value)
    ane_models = {row["model"] for row in rows if row["name"] == "ane.evaluate_api" and row["request"] >= 0}
    if type(expected_models) is not int or expected_models not in (1, 2):
        raise ValueError("explicit one-graph or joint two-graph contract required")
    if runtime and len(ane_models) != expected_models:
        raise ValueError("fixed runtime ANE model identity count differs from the explicit contract")
    if not runtime and ane_models:
        raise ValueError("GPU control unexpectedly executed ANE")
    summaries = []
    for request in runs:
        index = request["index"]
        owned = [row for row in rows if row["request"] == index]
        cb = [row for row in owned if row["name"] == "metal.command_buffer"]
        if not cb:
            raise ValueError("complete request lacks observed Metal command buffers")
        calls = Counter(row["name"] for row in owned)
        if runtime and not calls["ane.evaluate_api"]:
            raise ValueError("runtime request lacks actual Private evaluate submissions")
        gpu = [(max(row["gpu_begin_ns"], request["begin_ns"]), min(row["gpu_end_ns"], request["end_ns"]))
               for row in cb if row["gpu_begin_ns"] and row["gpu_begin_ns"] <= request["end_ns"] and row["gpu_end_ns"] >= request["begin_ns"]]
        if not gpu:
            raise ValueError("actual request lacks available completed GPU timestamps")
        queues = defaultdict(list)
        for row in cb:
            if row["gpu_begin_ns"]:
                queues[row["queue"]].append((row["gpu_begin_ns"], row["gpu_end_ns"]))
        spans = {name: sum(row["end_ns"] - row["begin_ns"] for row in owned if row["name"] == name) / 1e9 for name in calls}
        summaries.append(dict(index=index, api_calls=dict(calls), api_or_cb_lifetime_seconds=spans, observed_command_buffers=len(cb),
                              gpu_timestamp_unavailable=sum(not row["gpu_begin_ns"] for row in cb),
                              observed_gpu_window_union_seconds=union_ns(gpu) / 1e9,
                              per_queue_gpu_window_union_seconds={str(queue): union_ns(intervals) / 1e9 for queue, intervals in queues.items()}))
    hot = [item for item in summaries if item["index"] > 0]
    if runtime and expected_models == 2:
        # Match both identities to cumulative native submissions over every
        # request, including the FFN owner's three initial capability calls.
        # No broad relaxation of the existing single-graph contract.
        observed = sorted(tuple(sum(r["name"] == "ane.evaluate_api" and r["request"] == run["index"] and r["model"] == model
                                    for r in rows) for run in runs) for model in ane_models)
        expected = [[], []]
        previous_ffn = previous_qkv = 0
        for index, run in enumerate(runs):
            ffn = run["metrics"]["hybrid"]["runtime_calls_session_total"]
            qkv = run["metrics"]["qwen_private_qkv"]["predictions_session_total"]
            capability=run['metrics']['hybrid'].get('runtime_weight',{}).get('capability_calls_session_total',3)
            expected[0].append(ffn - previous_ffn + (capability if index == 0 else 0))
            expected[1].append(qkv - previous_qkv)
            previous_ffn, previous_qkv = ffn, qkv
        if observed != sorted(tuple(x) for x in expected):
            raise ValueError("joint graph actual per-model evaluations differ from native successful submissions")
    return dict(requests=summaries, runtime_model_count=len(ane_models),
                hot_no_compile_observed=all(not item["api_calls"].get("ane.compile_api", 0) for item in hot) if hot else None,
                hot_no_model_load_observed=all(not item["api_calls"].get("ane.load_api", 0) for item in hot) if hot else None,
                clock_calibration_passed=True,
                scope="observed CB window unions, not shader busy time or all GPU work; ANE spans are host API calls, not ANE hardware durations")


def native_deltas(runs):
    names = ("stage_seconds_session_total", "stage_wait_seconds_session_total", "async_ane_wait_seconds_session_total",
             "post_join_seconds_session_total", "pre_ffn_seconds_session_total", "hybrid_ffn_seconds_session_total",
             "lora_input_ready_seconds_session_total", "device_io_calls_session_total", "hybrid_blocks_session_total")
    previous = dict.fromkeys(names, 0)
    results = []
    for row in runs:
        current = (row["metrics"].get("hybrid") or {}).get("runtime_weight") or {}
        if not current:
            continue
        values = {name: current[name] for name in names}
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < previous[name] for name, value in values.items()):
            raise ValueError("runtime graph counters reset/regressed or are nonfinite")
        results.append(dict(index=row["index"], **{name: values[name] - previous[name] for name in names}))
        previous = values
    return results


def analyze_directory(root, output=None):
    root = Path(root)
    report = json.loads((root / "trial/report.json").read_text())
    if report.get("status") != "complete" or not report.get("runs"):
        raise ValueError("complete actual model process required")
    runs = report["runs"]
    heterogeneous = report.get("scenario") in ("sequence-matrix","text-prefix-matrix")
    result = dict(mode=report["mode"], route=report["route"], complete_requests=len(runs), qualification_passed=False,
                  first_request_seconds=runs[0]["request_wall_including_export"],
                  hot_request_seconds=[row["request_wall_including_export"] for row in runs[1:]],
                  hot_median_seconds=statistics.median(row["request_wall_including_export"] for row in runs[1:]) if len(runs) > 1 and not heterogeneous else None,
                  native_runtime_deltas=native_deltas(runs))
    if heterogeneous:
        if report.get("scenario")=="text-prefix-matrix":
            result["heterogeneous_text_prefix_matrix"]=True
            result["timing_comparability"]="Different prompts/prefix states; no pooled hot median or cross-prompt speedup denominator."
        else:
            result["heterogeneous_sequence_matrix"] = True
            result["timing_comparability"] = "Different tasks/reference counts; no pooled hot median or cross-task speedup denominator."
    result["phase_seconds"] = [dict(index=row["index"], full_request=row["request_wall_including_export"],
                                    **{key: value for key, value in row["metrics"]["timings_seconds"].items()
                                       if key in ("text_encode", "image_encode", "denoise", "vae_decode")}) for row in runs]
    result["prefix_snapshot_observations"] = [dict(index=row["index"],
        enabled="edit prefix KV snapshot " in row["metrics"]["acceleration_selection"],
        hit="edit prefix KV snapshot hit" in row["metrics"]["acceleration_selection"]) for row in runs]
    trace = []
    if report["mode"] != "stock":
        observed = json.loads((root / "trial/observer.json").read_text())
        result["observer"] = observer_contract(observed, runs, report["route"] != "gpu", expected_models=2 if report.get("private_qkv") else 1)
        epoch = min(row["begin_ns"] for row in observed["events"])
        for row in observed["events"]:
            trace.append(dict(name=row["name"], cat="cb-completion-lifetime" if row["name"] == "metal.command_buffer" else "host-api", ph="X", pid=1, tid=row["thread"],
                              ts=(row["begin_ns"] - epoch) / 1000, dur=(row["end_ns"] - row["begin_ns"]) / 1000,
                              args={key: row[key] for key in ("request", "model", "queue", "ok")}))
            if row["gpu_begin_ns"]:
                trace.append(dict(name="GPU command buffer", cat="metal-device-window", ph="X", pid=2, tid=row["queue"],
                                  ts=(row["gpu_begin_ns"] - epoch) / 1000, dur=(row["gpu_end_ns"] - row["gpu_begin_ns"]) / 1000,
                                  args=dict(request=row["request"], scope="CB elapsed window, not shader busy time")))
        for row in runs:
            trace.append(dict(name=f"request {row['index']}", cat="complete-request", ph="X", pid=1, tid=0,
                              ts=(row["begin_ns"] - epoch) / 1000, dur=(row["end_ns"] - row["begin_ns"]) / 1000))
        for row in report["progress"]:
            trace.append(dict(name="model progress", cat="model-progress", ph="i", s="t", pid=1, tid=0,
                              ts=(row["at_ns"] - epoch) / 1000, args=dict(request=row["request"], event=row["event"])))
    diagnostics = defaultdict(list)
    matrix_request = -1
    for line in (root / "stderr.log").read_text().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if "qwen21_gpu_op" in row:
            diagnostics[("gpu-op-layer0", row["phase"], row["qwen21_gpu_op"])].append(row["seconds"])
        elif "runtime_ane_layer" in row:
            diagnostics[("runtime-layer", str(row["runtime_ane_layer"]), row["axis"])].append(row["ffn_seconds"])
        elif "qwen21_step" in row:
            if heterogeneous and row["qwen21_step"] == 0:
                matrix_request += 1
            phase = f"request{matrix_request}:{row['phase']}" if heterogeneous else row["phase"]
            diagnostics[("sampler-step", phase, str(row["qwen21_step"]))].append(row["seconds"])
    result["diagnostic_groups"] = [dict(kind=key[0], phase_or_layer=key[1], name=key[2], samples=len(values), median_seconds=statistics.median(values))
                                    for key, values in sorted(diagnostics.items())]
    result["limitations"] = ["Instrumented modes are diagnostic, not primary performance samples.",
                              "Stage work may overlap GPU/ANE waits; do not add these totals or infer physical ANE overlap.",
                              "GPU-op diagnostics synchronize block0 and change graph execution; not whole-DiT operator attribution.",
                              "No per-kernel counters or attribution of other processes/devices; compile hooks cover the observed Private client selectors."]
    destination = Path(output) if output is not None else root
    if output is not None:
        destination.mkdir(parents=True, exist_ok=False)
    (destination / "trace.json").write_text(json.dumps(dict(traceEvents=trace, displayTimeUnit="ms")))
    (destination / "profile-analysis.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="fresh derived-artifact directory; preserve historical exports")
    args = parser.parse_args()
    print(json.dumps(analyze_directory(args.root, args.output), indent=2))
