"""Offline contracts for complete real-source private/shared Qwen FFN records.

No model, framework or candidate imports. Verify declared observations and ratios;
this does not reexecute hardware, reconstruct private outputs, or attest a run.
"""
import math
import random
import statistics
import struct

from common import report

ARMS = ("gpu", "copy-serial", "copy-hybrid", "shared-serial", "shared-hybrid")
BOUNDARY = "ready original BF16 GPU input to complete BF16 GPU output; GPU pack, copy control, evaluate, suffix, FP32 restore/join and MLX drain included"
IDENTITIES = {"request_input_matches", "request_output_matches", "metal_input_matches", "metal_output_matches", "input_output_distinct"}


def require(value, message):
    if not value:
        raise ValueError(message)


def bf16_rne(value):
    require(type(value) in (float, int) and math.isfinite(value), "finite FP32 join value required")
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("representable finite FP32 join value required") from error
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def cpu_join(tail_bf16, partial_fp16):
    require(all(type(value) is int and 0 <= value <= 65535 for value in (tail_bf16, partial_fp16)), "typed 16-bit source codes required")
    a = struct.unpack("<f", struct.pack("<I", tail_bf16 << 16))[0]
    b = struct.unpack("<e", struct.pack("<H", partial_fp16))[0]
    return bf16_rne(a + b)


def identity(value):
    require(set(value) == IDENTITIES and all(item is True for item in value.values()), "actual private/request/Metal identity proof missing")


def port(value):
    expected = {"Batches": 1, "Channels": 1, "Height": 1024, "Width": 4096, "RowStride": 8192,
                "PlaneStride": 8388608, "BatchStride": 8388608, "PlaneCount": 1, "Depth": 1}
    for key, item in expected.items():
        require(type(value.get(key)) is int and value[key] == item, "version-bound actual two-dimensional port differs")
    require(value.get("Type") == "Float16" and ("Offset" not in value or type(value["Offset"]) is int and value["Offset"] == 0), "explicit Float16/zero-offset feature required")


def metric(value):
    require(value.get("finite") is True, "finite complete/reference result required")
    for key in ("relative_l2", "cosine", "max_abs"):
        require(type(value.get(key)) in (int, float) and math.isfinite(value[key]), "typed finite metric required")
    require(0 <= value["relative_l2"] <= .03 and .999 <= value["cosine"] <= 1 + 1e-10 and value["max_abs"] >= 0, "fixed full/prefix/FP64 budget rejected")


def ratios(samples, repeats):
    measured = {i: {} for i in range(repeats)}
    for value in samples:
        if value["phase"] == "measured": measured[value["trial"]][value["arm"]] = value["wall_ns"]
    require(all(set(value) == set(ARMS) for value in measured.values()), "all five paired arms required")
    result = []
    for a, b in (("gpu", "copy-serial"), ("gpu", "copy-hybrid"), ("gpu", "shared-serial"), ("gpu", "shared-hybrid"),
                 ("copy-hybrid", "shared-hybrid"), ("shared-serial", "shared-hybrid")):
        values = [row[a] / row[b] for row in measured.values()]
        rng = random.Random(73); boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
        result.append({"baseline": a, "candidate": b, "paired_ratios": values, "median_paired_host_ratio": statistics.median(values),
                       "bootstrap_95pct": [boot[49], boot[1949]], "production_qualified": False})
    return result


def verify(document):
    require(document.get("schema") == "tc-qwen-shared-probe-v1" and document.get("status") == "measured" and document.get("mode") == "benchmark"
            and document.get("device_requested") is True and type(document.get("native_returncode")) is int and document["native_returncode"] == 0
            and document.get("errors") == [] and all(document.get(key) is True for key in
                ("source_inputs_unchanged", "cloned_graph_unchanged", "original_spans_unchanged", "mlx_assets_unchanged")), "complete source-stable executed benchmark required")
    native = document["native"]
    for value in (document, native):
        require(value.get("production_qualified") is False and value.get("physical_overlap_qualified") is False
                and value.get("runtime_internal_copy_behavior") == "unknown", "no physical/internal-copy/product qualification granted")
    require(native.get("schema") == "tc-qwen-shared-native-v1" and native.get("status") == "measured" and native.get("errors") == []
            and native.get("boundary") == BOUNDARY and native.get("mlx_version") == "0.32.0" and native.get("gpu_compile_enabled") is True
            and native.get("gates") == {"relative_l2_max": .03, "cosine_min": .999} and native.get("pre_arm_validation_drain") is True,
            "complete compiled BF16 GPU denominator and fixed timing/quality boundary required")
    require(document["shape_mhf"] == native["shape_mhf"] == [1024,4096,12288] and all(type(x) is int for x in document["shape_mhf"] + native["shape_mhf"])
            and type(document["share"]) is int and type(native["share"]) is int and document["share"] == native["share"] == 2048
            and type(document["source_layer"]) is int and document["source_layer"] == 0 and document["loras"] == [] and native["logical_gpu_weight_bytes"] == 603979776
            and native["logical_private_weight_bytes"] == 50331648, "full original layer0 workload and representation costs required")
    port(native["input_descriptor"]); port(native["output_descriptor"])
    for key in ("unload_succeeded", "restore_deleter_pointer_matches", "owned_mappings_released_after_completion",
                "restore_views_alias_actual_buffers", "compiled_join_uses_runtime_partial_argument"):
        require(native.get(key) is True, "actual restore alias/lifecycle proof incomplete")
    require(type(native["restore_adoption_releases"]) is int and native["restore_adoption_releases"] == 2
            and native["mapping_quarantined_until_child_exit"] is False and native["mapping_reuse_after_failure"] is False, "successful normal owner lifecycle required")
    identity(native["initial_identity"]); identity(native["final_identity"])
    if "compile_selector_called" in native:
        require(type(native["compile_selector_called"]) is bool, "typed compile-selector observation required")
        if native["compile_selector_called"] is False:
            require(native["compile_wall_ns"] is None and document["admission"] == {"kind": "read-only-prepared-load", "minimum_free_mib": 128}
                    and document.get("prepared_source_report_sha256") and document.get("prepared_source_manifest_sha256"), "load-only must retain source lineage and null uncalled compile duration")
    cycle, repeats = native["point_cycle"], native["measured_groups"]
    require(isinstance(cycle, list) and len(cycle) in (2,3) and len(set(cycle)) == len(cycle)
            and set(cycle) <= {"layer-0-step-1", "layer-0-step-10", "layer-0-step-19"}
            and type(repeats) is int and repeats in (10,30), "alternate explicit complete actual timesteps and bounded balanced groups required")
    expected = []
    for group in range(repeats + 3):
        phase = "first" if group == 0 else "warmup" if group < 3 else "measured"
        trial = 0 if group == 0 else group - 1 if group < 3 else group - 3
        order = group if group < 3 else trial
        for position in range(5):
            index = (order % 5 + (4 - position if order // 5 % 2 else position)) % 5
            expected.append((phase,trial,ARMS[index],cycle[group % len(cycle)]))
    samples = native["samples"]
    require([(row["phase"],row["trial"],row["arm"],row["point"]) for row in samples] == expected, "all first/warmup/measured calls, actual-point cycle and ten-row order required")
    exact = native.get("full_exact_runtime_join_witness", False)
    require(type(exact) is bool, "typed join-witness protocol revision required")
    joins, stale, previous_point, intersections = 0, 0, None, []
    for row in samples:
        require(type(row["trial"]) is int and type(row["wall_ns"]) is int and row["wall_ns"] > 0 and row["output_dtype"] == "bfloat16"
                and row["numeric_pass"] is True and row["tail_guards_pass"] is True and row["physical_overlap_qualified"] is False
                and row["runtime_internal_copy_behavior"] == "unknown", "typed full consumer/gate/completion record required")
        gpu, copied, hybrid = row["arm"] == "gpu", row["arm"].startswith("copy"), row["arm"].endswith("hybrid")
        keys = {"full_original", "independent_full_sample"} | (set() if gpu else {"prefix_original", "independent_prefix_sample"})
        require(set(row["metrics"]) == keys, "full/prefix and independent complete-K/F sample gates required")
        for value in row["metrics"].values(): metric(value)
        if gpu:
            require(row["metrics"]["full_original"]["relative_l2"] == row["metrics"]["full_original"]["max_abs"] == 0 and row["host_events"] == [], "full compiled BF16 GPU must exactly match saved replica")
        require(type(row["external_feature_copy_count"]) is int and row["external_feature_copy_count"] == (2 if copied else 0)
                and type(row["external_feature_copy_bytes"]) is int and row["external_feature_copy_bytes"] == (16777216 if copied else 0), "same-client FP16 copy control accounting differs")
        if gpu: continue
        identity(row["identity_before"]); identity(row["identity_after"])
        required = {"gpu-bf16-to-fp16-pack-complete", "private-evaluate-returned", "gpu-suffix-output-ready", "gpu-fp32-restore-join-bf16-drain"}
        if copied: required |= {"cpu-input-feature-copy", "cpu-output-feature-copy"}
        require(len(row["host_events"]) == len(required) and {event["name"] for event in row["host_events"]} == required, "whole pack/copy/evaluate/suffix/restore/join boundary required")
        table = {}; last = -1
        for event in row["host_events"]:
            start, duration = event["start_ns"], event["duration_ns"]
            require(type(start) is int and type(duration) is int and 0 <= start and start >= last and duration > 0
                    and start + duration <= row["wall_ns"] and event["clock_domain"] == "same-process-uptime-raw-host", "same-process host clock and full interval association required")
            table[event["name"]] = (start,start+duration); last = start
        pack, private, suffix, join = (table[name] for name in ("gpu-bf16-to-fp16-pack-complete", "private-evaluate-returned", "gpu-suffix-output-ready", "gpu-fp32-restore-join-bf16-drain"))
        producer_end = table["cpu-input-feature-copy"][1] if copied else pack[1]
        require(producer_end <= min(private[0],suffix[0]) and join[0] >= max(private[1],suffix[1])
                and (hybrid or private[1] <= suffix[0]), "producer/serial/consumer dependency violated")
        if copied:
            require(pack[1] <= table["cpu-input-feature-copy"][0] and private[1] <= table["cpu-output-feature-copy"][0]
                    and table["cpu-output-feature-copy"][1] <= join[0], "feature copies cannot bypass producers")
        intersections.append({"phase":row["phase"],"trial":row["trial"],"arm":row["arm"],"point":row["point"],
            "api_host_intersection_ns":max(0,min(private[1],suffix[1])-max(private[0],suffix[0])),"physical_overlap_qualified":False})
        if exact:
            require("exact_runtime_join" in row, "new join proof cannot be backfilled into an older sample")
            value = row["exact_runtime_join"]
            require(type(value["elements"]) is int and value["elements"] == 4194304 and type(value["exact_mismatches"]) is int
                    and value["exact_mismatches"] == 0 and value["expected_bf16_sha256"] == value["actual_bf16_sha256"]
                    and isinstance(value["actual_bf16_sha256"],str) and len(value["actual_bf16_sha256"]) == 64
                    and all(char in "0123456789abcdef" for char in value["actual_bf16_sha256"])
                    and value["reference_kind"] == "independent CPU FP32 add and explicit BF16 RNE from actual tail/private outputs", "whole actual-part runtime join binding failed")
            joins += 1
            if row["arm"] == "shared-hybrid":
                if previous_point is not None:
                    value = row["stale_actual_join_control"]
                    require(value["previous_actual_point"] == previous_point and value["expected_point"] == row["point"]
                            and type(value["mismatched_elements"]) is int and 0 < value["mismatched_elements"] <= 4194304
                            and value["used_for_speed_ratio"] is False and value["new_device_dispatch_performed"] is False, "previous actual point's output must reject the new actual join")
                    stale += 1
                previous_point = row["point"]
        else:
            require("exact_runtime_join" not in row and "stale_actual_join_control" not in row, "do not backfill a newer proof into older samples")
    result = ratios(samples,repeats)
    require(document["comparisons"] == result, "paired ratios or uncertainty arithmetic differs")
    return {"complete_consumers":len(samples),"exact_runtime_join_consumers":joins,"stale_actual_join_observations":stale,
            "median_host_wall_ns":{arm:statistics.median(row["wall_ns"] for row in samples if row["arm"] == arm and row["phase"] == "measured") for arm in ARMS},
            "host_intersections":intersections,"physical_overlap_qualified":False,"production_qualified":False,
            "raw_private_outputs_reexecuted_by_verifier":False,"runtime_internal_copy_behavior":"unknown"}


def join_contract(seed=17):
    assert cpu_join(0x3f80,0x3c00) == 0x4000
    # FP32 1+2^-8 is a BF16 halfway case; retain the even low bit.
    assert bf16_rne(1+2**-8) == 0x3f80 and bf16_rne(1+3*2**-8) == 0x3f82
    return report("real-shared-join", {"actual_private_port_nchw":[1,1,1024,4096],"independent_cpu_bf16_tie_cases":True,
                  "ten_row_five_arm_order_is_not_a_physical_overlap_trace":True,"actual_device_executed":False,"seed":seed},
                  scope="CPU rounding/actual-port and complete-consumer contracts, not a live model benchmark")
