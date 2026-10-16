"""Independent stdlib witnesses for the synthetic MLX adoption/owner experiment.

The native producer writes every logical element. Only THIS verifier exploits
the declared period-37 synthetic recipe to hash large arrays with bounded RAM.
No MLX import, device execution, real-model or performance qualification.
"""
import hashlib
import math
from pathlib import PurePosixPath
import struct

from common import report

ALIASES = {"buffer_contents_matches_surface", "allocator_raw_pointer_matches_surface", "same_metal_device", "shared_storage",
           "zero_view_offset", "buffer_covers_surface", "mlx_buffer_hazard_tracking_untracked"}
CASES = {"padded": (32, 33, 64), "port": (1024, 4096, 4096)}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest_pattern(values, dtype, count):
    pattern = struct.pack("<" + dtype * len(values), *values)
    result = hashlib.sha256()
    repeats, remainder = divmod(count, len(values))
    while repeats:
        chunk = min(repeats, 4096)
        result.update(pattern * chunk); repeats -= chunk
    result.update(struct.pack("<" + dtype * remainder, *values[:remainder]))
    return result.hexdigest()


def witness(rows, cols, generation, seed):
    require(type(rows) is int and type(cols) is int and (rows, cols) in ((32, 33), (1024, 4096)), "bounded complete logical shape required")
    require(type(generation) is int and 0 <= generation <= 2 and type(seed) is int and 0 <= seed <= 10000, "bounded exact generation/seed required")
    values = [((i * 7 + generation * 5 + seed * 11) % 37 - 18) / 16. for i in range(37)]
    return {"input_sha256": digest_pattern(values, "e", rows * cols),
            "consumer_sha256": digest_pattern([2 * value + .25 for value in values], "f", rows * cols)}


def alias(value):
    require(set(value) == ALIASES and all(item is True for item in value.values()), "actual MLX alias/storage/hazard proof differs")


def metric(value, digest, mismatches=0, maximum=0):
    require(type(value.get("mismatched_elements")) is int and type(value.get("nonfinite_elements")) is int
            and type(value.get("maximum_finite_abs_error")) in (int, float), "typed exact metric required")
    require(value == {"mismatched_elements": mismatches, "nonfinite_elements": 0, "maximum_finite_abs_error": maximum,
                      "consumer_sha256": digest}, "independent complete consumer witness differs")


def envelope(document):
    require(document.get("schema") == "tc-mlx-adoption-probe-v1" and document.get("device_requested") is True
            and document.get("source_inputs_unchanged") is True and document.get("mlx_assets_unchanged") is True
            and type(document.get("build_returncode")) is int and document["build_returncode"] == 0 and document.get("errors") == [],
            "explicit source/asset-stable executed client required")
    native = document["native"]
    require(native.get("schema") == "tc-mlx-adoption-native-v1" and native.get("mlx_version") == "0.32.0", "version-bound native MLX contract required")
    for value in (document, native):
        require(value.get("synthetic") is True and value.get("real_qwen") is False and value.get("formal_performance_score") is False
                and value.get("production_qualified") is False and value.get("physical_overlap_qualified") is False
                and value.get("runtime_internal_copy_behavior") == "unknown", "synthetic adoption cannot grant model/physical/performance qualification")
    require(document["case"] in CASES and native["case"] == document["case"] and type(document["seed"]) is int
            and type(native["seed"]) is int and native["seed"] == document["seed"], "declared case/seed differs")
    rows, cols, pitch = CASES[document["case"]]
    require(native["logical_shape"] == [rows, cols] and all(type(x) is int for x in native["logical_shape"])
            and type(native["pitch_elements"]) is int and native["pitch_elements"] == pitch
            and type(native["payload_bytes"]) is int and native["payload_bytes"] == rows * pitch * 2, "physical/logical byte mapping differs")
    page, size = native["page_bytes"], native["allocation_bytes"]
    require(type(page) is int and 4096 <= page <= 65536 and page & (page - 1) == 0
            and type(size) is int and size % page == 0 and size >= rows * pitch * 2 + page, "whole-page region and tail guard required")
    require(isinstance(document.get("macos_build"), str) and document["macos_build"] and document.get("sdk_version")
            and document.get("compiler_version"), "captured build/SDK/compiler identity required")
    headers = document["mlx_assets"]["headers"]
    names = [item["path"] for item in headers]
    require(all(isinstance(name, str) and name.startswith("include/") and ".." not in name.split("/") for name in names), "bounded installed-header paths required")
    require(names == sorted(set(names), key=lambda name: PurePosixPath(name).parts), "explicit unique ordered installed-header manifest required")
    require(set(document["mlx_assets"]["libraries"]) == {"libmlx.dylib", "libjaccl.dylib", "mlx.metallib"}, "all three selected library assets required")
    return native, rows, cols, pitch


def sample(value, rows, cols, seed, generation):
    require(type(value["generation"]) is int and value["generation"] == generation, "ordered actual generations required")
    expected = witness(rows, cols, generation, seed)
    require(value["input_sha256"] == expected["input_sha256"], "full logical GPU-produced input differs")
    metric(value["consumer"], expected["consumer_sha256"]); alias(value["actual_alias"])
    require(all(value.get(key) is True for key in ("tail_guard_pass", "owned_producer_completed", "mlx_stream_drained"))
            and type(value["owner_deleter_calls"]) is int and value["owner_deleter_calls"] == 0
            and value.get("formal_performance_score") is False, "producer/consumer/owner lifetime gate failed")
    require(type(value["diagnostic_boundary_ns"]) is int and value["diagnostic_boundary_ns"] > 0, "positive diagnostic span required")
    events = value["host_events"]
    require([event["name"] for event in events] == ["mlx-drain-before-external-write", "external-metal-producer-complete", "new-mlx-consumer-complete"], "both sides of external untracked write must be completed")
    end = 0
    for event in events:
        start, duration = event["start_ns"], event["duration_ns"]
        require(type(start) is int and type(duration) is int and end <= start and duration > 0
                and start + duration <= value["diagnostic_boundary_ns"] and event["clock_domain"] == "same-process-uptime-raw-host", "serial same-clock completion dependency differs")
        end = start + duration


def verify(document):
    native, rows, cols, pitch = envelope(document)
    require(document["status"] == native["status"] == "actual-adoption-and-lifetime-validated"
            and type(document["native_returncode"]) is int and document["native_returncode"] == 0 and native["errors"] == []
            and native["surface_released_after_owned_completions"] is True and native["mapping_quarantined_until_child_exit"] is False
            and native["mapping_reuse_after_failure"] is False, "complete successful owned lifecycle required")
    require([record["mode"] for record in native["records"]] == ["raw-pointer", "allocator-buffer"], "both import contracts required")
    for record in native["records"]:
        require(record["status"] == native["status"], "complete per-mode protocol required")
        for key in ("flat_alias", "view_alias", "same_dtype_copy_alias"):
            alias(record[key])
        for key, expected in (("deleter_calls_after_construct", 0), ("deleter_calls_after_flat_scope", 0),
                              ("deleter_calls_before_last_view_release", 0), ("deleter_calls_after_last_view_release", 1)):
            require(type(record[key]) is int and record[key] == expected, "view ownership/deleter ordering differs")
        require(record["deleter_pointer_matches_surface"] is True and record["released_after_owned_completions"] is True
                and record["snapshot_kind"] == "explicit-FP16-to-FP32-materialization" and record["snapshot_has_distinct_backing"] is True,
                "separate materialization or completion-bound release missing")
        require(len(record["samples"]) == 3, "all three generations required")
        for generation, value in enumerate(record["samples"]):
            sample(value, rows, cols, document["seed"], generation)
        metric(record["new_consumer_of_same_dtype_copy"], witness(rows, cols, 1, document["seed"])["consumer_sha256"])
        stale = record["stale_controls"]
        require(type(stale["actual_cached_generation"]) is int and stale["actual_cached_generation"] == 0
                and type(stale["expected_generation"]) is int and stale["expected_generation"] == 1 and stale["used_for_speed_ratio"] is False, "explicit stale-result distinction required")
        for key in ("cached_evaluated_output", "new_consumer_of_materialized_snapshot"):
            metric(stale[key], witness(rows, cols, 0, document["seed"])["consumer_sha256"], rows * cols, 4.)
        if pitch != cols:
            wrong = record["packed_stride_control"]
            require(type(wrong["mismatched_elements"]) is int and type(wrong["nonfinite_elements"]) is int
                    and 0 <= wrong["nonfinite_elements"] <= wrong["mismatched_elements"] <= rows * cols and wrong["mismatched_elements"] > 0
                    and type(wrong["maximum_finite_abs_error"]) in (int, float) and math.isfinite(wrong["maximum_finite_abs_error"]), "actual packed-stride negative required")
        else:
            require("packed_stride_control" not in record, "no padded-stride negative fabricated for contiguous port")
    return {"main_generation_consumers": 6, "same_dtype_copy_consumers": 2, "full_positive_consumer_hashes": 8,
            "cached_result_rejections": 2, "materialized_snapshot_rejections": 2, "packed_stride_rejections": 2 if pitch != cols else 0,
            "logical_elements_per_consumer": rows * cols, "formal_performance_score": False, "real_qwen": False,
            "physical_overlap_qualified": False, "runtime_internal_copy_behavior": "unknown"}


def verify_failure(document):
    native, rows, cols, _ = envelope(document)
    require(document["status"] == "native-failed" and type(document["native_returncode"]) is int and document["native_returncode"] == 1
            and native["status"] == "failed" and native["errors"] == ["explicit copy did not create distinct backing"]
            and native["mapping_quarantined_until_child_exit"] is True and native["surface_released_after_owned_completions"] is False
            and native["mapping_reuse_after_failure"] is False and len(native["records"]) == 1, "retain exact client snapshot-assumption failure, not hardware unsupported")
    record = native["records"][0]
    require(record["mode"] == "raw-pointer" and record["snapshot_has_distinct_backing"] is False and len(record["samples"]) == 1,
            "failed protocol must not be promoted to all-generation adoption")
    alias(record["flat_alias"]); alias(record["view_alias"])
    sample(record["samples"][0], rows, cols, document["seed"], 0)
    return {"partial_full_consumer_hashes": 1, "complete_protocol_passed": False,
            "failure_kind": "client incorrectly assumed same-dtype Copy was an independent snapshot", "hardware_unsupported": False}


def adoption_contract(seed=17):
    first, next_value = witness(32, 33, 0, seed), witness(32, 33, 1, seed)
    assert first != next_value
    return report("mlx-adoption-contract", {"whole_page_import_then_strided_view": [32, 33, 64],
                  "same_dtype_copy_is_not_an_independent_snapshot_in_the_bound_source": True,
                  "cached_evaluated_result_is_not_external_generation_tracking": True,
                  "runtime_internal_copy_behavior": "unknown", "actual_device_executed": False},
                  scope="CPU synthetic full-hash/owner/copy contracts, not a live MLX or model test")
