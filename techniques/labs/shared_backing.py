"""Independent CPU witnesses for a bounded Metal/private-ANE surface experiment.

Matching external backing is not a claim about runtime-internal copies, physical
placement, overlap, or performance of a real model. No device modules imported.
"""
import hashlib
import math
import statistics
import struct

from common import report

BOUNDARY = "GPU producer encode through GPU consumer completion; feature copies included for copy control; validation excluded"
IDENTITY_KEYS = {"request_input_surface_matches", "request_output_surface_matches", "input_metal_contents_matches",
                 "output_metal_contents_matches", "input_output_distinct"}


def require(value, message):
    if not value:
        raise ValueError(message)


def source_values(channels, width, generation, seed):
    require(type(channels) is int and channels in (32, 64) and type(width) is int and width in (33, 65), "bounded shape required")
    require(type(generation) is int and 0 <= generation <= 27 and type(seed) is int and 0 <= seed <= 10000, "bounded generation/seed required")
    return [((c * 7 + x * 3 + generation * 5 + seed * 11) % 37 - 18) / 16.
            for c in range(channels) for x in range(width)]


def witness(channels, width, generation, seed):
    values = source_values(channels, width, generation, seed)
    return {name: hashlib.sha256(struct.pack("<" + dtype * len(data), *data)).hexdigest()
            for name, dtype, data in (("input_sha256", "e", values),
                                      ("ane_output_sha256", "e", [2 * x + .25 for x in values]),
                                      ("consumer_sha256", "f", [6 * x + .25 for x in values]))}


def physical_layout(value, channels, width):
    for key, expected in (("Batches", 1), ("Channels", channels), ("Height", 1), ("Width", width)):
        require(type(value.get(key)) is int and value[key] == expected, "explicit physical NCHW required")
    require(value.get("Type") == "Float16", "explicit Float16 port required")
    for key, expected in (("Offset", 0), ("Depth", 1), ("PlaneCount", channels)):
        if key in value:
            require(type(value[key]) is int and value[key] == expected, "unsupported offset/depth/planes")
    for key in ("RowStride", "PlaneStride", "BatchStride"):
        require(type(value.get(key)) is int and 0 < value[key] <= 8 * 1024**2, "bounded physical pitch required")
    row, plane, batch = (value[key] for key in ("RowStride", "PlaneStride", "BatchStride"))
    require(row >= 2 * width and plane >= row and plane % 2 == 0 and batch >= channels * plane, "invalid byte pitch")
    return batch


def identity(value):
    require(set(value) == IDENTITY_KEYS and all(item is True for item in value.values()), "actual request/Metal backing proof incomplete")


def verify(document):
    require(document.get("schema") == "tc-private-shared-probe-v1" and document.get("status") == "backing-and-consumer-validated"
            and document.get("device_requested") is True and type(document.get("native_returncode")) is int and document["native_returncode"] == 0
            and document.get("source_inputs_unchanged") is True and document.get("model_source_unchanged") is True
            and document.get("boundary") == BOUNDARY and document.get("errors") == [], "source-stable explicit executed probe required")
    native = document["native"]
    require(native.get("schema") == "tc-private-shared-native-v1" and native.get("status") == "backing-and-consumer-validated"
            and native.get("unload_succeeded") is True and native.get("mapping_released_after_owned_completion_and_unload") is True
            and native.get("mapping_quarantined_until_child_exit") is False and native.get("mapping_reuse_after_failure") is False
            and native.get("errors") == [] and native.get("formal_performance_score") is False, "completed ownership lifecycle required")
    for value in (document, native):
        require(value.get("production_qualified") is False and value.get("physical_overlap_qualified") is False
                and value.get("runtime_internal_copy_behavior") == "unknown" and value.get("full_system_zero_copy", False) is False,
                "external aliases cannot grant internal-copy/physical/model qualification")
    shape = document["shape_nchw"]
    require(isinstance(shape, list) and len(shape) == 4 and all(type(x) is int for x in shape)
            and shape[0] == shape[2] == 1 and shape == native["shape_nchw"] and all(type(x) is int for x in native["shape_nchw"]), "exact shape identity required")
    _, channels, _, width = shape
    seed, repeats = document["seed"], document["measured_groups"]
    require(type(repeats) is int and repeats in (8, 24) and type(native["measured_groups"]) is int and native["measured_groups"] == repeats
            and type(native["seed"]) is int and native["seed"] == seed, "declared paired workload differs")
    source_values(channels, width, 0, seed)
    counts = [physical_layout(native[key], channels, width) for key in ("input_descriptor", "output_descriptor")]
    require(native["output_descriptor"]["PlaneStride"] > width * 2, "padding witness required")
    page, allocations = native["page_bytes"], native["surface_allocation_bytes"]
    require(type(page) is int and 4096 <= page <= 65536 and page & (page - 1) == 0 and isinstance(allocations, list)
            and len(allocations) == 2 and all(type(size) is int and size % page == 0 and size >= count + page
                                           for size, count in zip(allocations, counts)), "page-aligned payload plus tail guard required")
    identity(native["initial_identity"]); identity(native["final_identity"])
    expected_order = [("first" if group == 0 else "warmup" if group < 3 else "measured",
                       0 if group == 0 else group - 1 if group < 3 else group - 3,
                       "copy" if (group + position) % 2 else "shared", group)
                      for group in range(repeats + 3) for position in range(2)]
    samples = native["samples"]
    require([(row["phase"], row["trial"], row["arm"], row["generation"]) for row in samples] == expected_order,
            "all first/warmup/measured calls and balanced order required")
    measured = {trial: {} for trial in range(repeats)}
    event_spans = {"shared": {}, "copy": {}}
    for row in samples:
        require(type(row["trial"]) is int and type(row["generation"]) is int and type(row["wall_ns"]) is int and row["wall_ns"] > 0,
                "typed positive timing/generation required")
        actual = witness(channels, width, row["generation"], seed)
        require(row["input_sha256"] == actual["input_sha256"] and row["ane_output_sha256"] == actual["ane_output_sha256"], "full logical surface hash differs from independent math")
        consumer = row["consumer"]
        require(type(consumer.get("mismatched_elements")) is int and type(consumer.get("nonfinite_elements")) is int
                and type(consumer.get("maximum_finite_abs_error")) in (int, float), "typed consumer metrics required")
        require(consumer == {"mismatched_elements": 0, "nonfinite_elements": 0, "maximum_finite_abs_error": 0,
                             "consumer_sha256": actual["consumer_sha256"]}, "full GPU consumer hash/exact gate differs")
        identity(row["identity_before"]); identity(row["identity_after"])
        require(all(row.get(key) is True for key in ("tail_guards_pass", "gpu_completions_checked", "output_poison_overwritten")), "guard/completion/poison proof missing")
        require(row.get("physical_overlap_qualified") is False and row.get("runtime_internal_copy_behavior") == "unknown", "physical/internal behavior not qualified")
        copied = row["arm"] == "copy"
        require(type(row["external_feature_copy_count"]) is int and row["external_feature_copy_count"] == (2 if copied else 0)
                and type(row["external_feature_copy_bytes"]) is int and row["external_feature_copy_bytes"] == (sum(counts) if copied else 0), "explicit external feature-copy accounting differs")
        names = ["gpu-producer-complete"] + (["cpu-input-feature-copy"] if copied else []) + ["private-evaluate-returned"]
        names += (["cpu-output-feature-copy"] if copied else []) + ["gpu-consumer-complete"]
        require([event["name"] for event in row["host_events"]] == names, "producer/private/consumer and copy events required")
        end = 0
        for event in row["host_events"]:
            start, duration = event["start_ns"], event["duration_ns"]
            require(type(start) is int and type(duration) is int and duration > 0 and end <= start
                    and start + duration <= row["wall_ns"] and event["clock_domain"] == "same-process-uptime-raw-host", "serial same-clock host dependency violated")
            end = start + duration
            if row["phase"] == "measured":
                event_spans[row["arm"]].setdefault(event["name"], []).append(duration)
        if row["phase"] == "measured":
            measured[row["trial"]][row["arm"]] = row["wall_ns"]
    negative = native["negative_controls"]
    require(all(type(negative.get(key)) is int for key in ("last_actual_generation", "stale_expected_generation", "packed_plane_bytes", "actual_plane_bytes")), "typed negative workload required")
    require(negative["last_actual_generation"] == repeats + 2 and negative["stale_expected_generation"] == repeats + 3
            and negative["packed_plane_bytes"] == width * 2 and negative["actual_plane_bytes"] == native["output_descriptor"]["PlaneStride"]
            and negative["consumer_tail_guard_pass"] is True and negative["used_for_speed_ratio"] is False, "bounded stride/stale controls required")
    old, new = (source_values(channels, width, generation, seed) for generation in (repeats + 2, repeats + 3))
    require(type(negative["stale_generation"].get("mismatched_elements")) is int and type(negative["stale_generation"].get("nonfinite_elements")) is int
            and type(negative["stale_generation"].get("maximum_finite_abs_error")) in (int, float), "typed stale metrics required")
    require(negative["stale_generation"] == {"mismatched_elements": channels * width, "nonfinite_elements": 0,
                "maximum_finite_abs_error": max(abs(6 * a - 6 * b) for a, b in zip(old, new)),
                "consumer_sha256": witness(channels, width, repeats + 2, seed)["consumer_sha256"]}, "actual stale output must be rejected against next generation")
    wrong = negative["packed_stride"]
    require(type(wrong["mismatched_elements"]) is int and type(wrong["nonfinite_elements"]) is int
            and 0 <= wrong["nonfinite_elements"] <= wrong["mismatched_elements"] <= channels * width and wrong["mismatched_elements"] > 0
            and type(wrong["maximum_finite_abs_error"]) in (int, float) and math.isfinite(wrong["maximum_finite_abs_error"])
            and wrong["maximum_finite_abs_error"] >= 0, "actual packed-stride interpretation must fail full consumer gate")
    return {"calls_independently_witnessed": len(samples), "negative_controls": 2, "logical_elements_per_call": channels * width,
            "median_host_wall_ns": {arm: statistics.median(row[arm] for row in measured.values()) for arm in ("shared", "copy")},
            "median_host_event_ns": {arm: {key: statistics.median(values) for key, values in events.items()} for arm, events in event_spans.items()},
            "diagnostic_median_paired_copy_over_shared": statistics.median(row["copy"] / row["shared"] for row in measured.values()),
            "formal_performance_score": False, "gpu_only_baseline_measured": False,
            "runtime_internal_copy_behavior": "unknown", "physical_overlap_qualified": False, "production_qualified": False}


def backing_proof(seed=17):
    physical = {"Batches": 1, "Channels": 32, "Height": 1, "Width": 33, "Type": "Float16",
                "RowStride": 128, "PlaneStride": 128, "BatchStride": 4096}
    assert physical_layout(physical, 32, 33) == 4096
    a, b = witness(32, 33, 0, seed), witness(32, 33, 1, seed)
    assert a["consumer_sha256"] != b["consumer_sha256"]
    identity({key: True for key in IDENTITY_KEYS})
    return report("backing-proof", {"physical_plane_bytes": 128, "incorrect_packed_plane_bytes": 66,
                  "changed_generation_has_distinct_consumer_hash": True, "external_identity_fields": sorted(IDENTITY_KEYS),
                  "runtime_internal_copy_behavior": "unknown", "actual_device_executed": False},
                  scope="CPU exact-dyadic and external-backing proof contracts, not a device benchmark")
