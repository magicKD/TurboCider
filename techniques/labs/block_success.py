"""Successful raw-pair block contracts, separate from frozen failure evidence.

No candidate/framework imports. Retained FFN input/reference are NOT the nine
ordinary-block arguments or actual private outputs. Local component observations
never qualify a full request, image, LoRA profile or physical device overlap.
"""
import hashlib
import json
from pathlib import Path
import random
import statistics
import struct

from block_context import ARMS, identity, metric, require
from real_shared_ffn import port

ELEMENTS = 4194304
PAYLOAD_BYTES = ELEMENTS * 2
RECIPE = "ordinary-block-exposed-ffn-v1"


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate raw artifact header key")
        result[key] = value
    return result


def raw_bf16(path, key):
    """Read an exact fresh 8MiB BF16 tensor; return public metadata and private bytes."""
    require(key in ("input", "reference"), "explicit input/reference key required")
    path = Path(path)
    require(path.is_file() and PAYLOAD_BYTES + 10 <= path.stat().st_size <= PAYLOAD_BYTES + 65544,
            "complete bounded raw BF16 file required")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "complete safetensors prefix required")
        size = struct.unpack("<Q", prefix)[0]
        require(2 <= size <= 65536 and size % 8 == 0, "bounded aligned raw tensor header required")
        header_bytes = stream.read(size)
        require(len(header_bytes) == size, "complete raw tensor header required")
        header = json.loads(header_bytes, object_pairs_hook=unique_object)
        require(isinstance(header, dict) and set(header) == {key, "__metadata__"}, "one explicit tensor and recipe only")
        node = header[key]
        require(isinstance(node, dict) and node == {"dtype": "BF16", "shape": [1,1024,4096], "data_offsets": [0,PAYLOAD_BYTES]}
                and all(type(v) is int for v in node["shape"] + node["data_offsets"])
                and header["__metadata__"] == {"recipe": RECIPE}, "exact shape/dtype/offset/recipe required")
        data = stream.read(PAYLOAD_BYTES + 1)
        require(len(data) == PAYLOAD_BYTES and path.stat().st_size == 8 + size + PAYLOAD_BYTES, "raw tensor trailing/short payload rejected")
    require(not any(code & 0x7f80 == 0x7f80 for (code,) in struct.iter_unpack("<H", data)), "nonfinite raw BF16 source rejected")
    return {"dtype": "BF16", "shape": [1,1024,4096], "recipe": RECIPE, "payload_bytes": len(data),
            "payload_sha256": hashlib.sha256(data).hexdigest(), "file_bytes": 8 + size + len(data),
            "file_sha256": hashlib.sha256(prefix + header_bytes + data).hexdigest(), "all_finite": True}, data


def expected_order(repeats):
    require(type(repeats) is int and repeats in (10,30), "bounded balanced groups required")
    result = []
    for group in range(repeats + 3):
        phase = "first" if group == 0 else "warmup" if group < 3 else "measured"
        trial = 0 if group == 0 else group - 1 if group < 3 else group - 3
        row = group if group < 3 else trial
        result.extend((phase, trial, ARMS[(row % 5 + (4 - position if row // 5 % 2 else position)) % 5]) for position in range(5))
    return result


def comparisons(samples, repeats):
    rows = {trial: {s["arm"]: s["wall_ns"] for s in samples if s["phase"] == "measured" and s["trial"] == trial} for trial in range(repeats)}
    require(all(set(row) == set(ARMS) for row in rows.values()), "all five same-input measured arms required")
    result = []
    for baseline, candidate in (("gpu-original","gpu-staged"), ("gpu-original","copy-hybrid"),
                                ("gpu-original","shared-serial"), ("gpu-original","shared-hybrid"),
                                ("gpu-staged","shared-hybrid"), ("copy-hybrid","shared-hybrid")):
        values = [row[baseline] / row[candidate] for row in rows.values()]
        rng = random.Random(73)
        boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
        result.append({"baseline": baseline, "candidate": candidate, "paired_ratios": values,
                       "median_paired_host_ratio": statistics.median(values), "bootstrap_95pct": [boot[49],boot[1949]], "production_qualified": False})
    return result


def verify(document, artifacts):
    require(document.get("schema") == "tc-qwen-ordinary-block-wrapper-v1" and document.get("status") == "measured"
            and document.get("device_requested") is True and type(document.get("native_returncode")) is int
            and document["native_returncode"] == 0 and document.get("errors") == [], "actual completed successful protocol required")
    require(type(document["seed"]) is int and 0 <= document["seed"] < 2**32 and document["seed"] != document["baseline_seed"]
            and is_sha(document["prompt_sha256"]) and document["prompt_sha256"] != document["baseline_prompt_sha256"], "new actual seed AND prompt required")
    for key in ("source_inputs_unchanged", "native_library_unchanged", "mlx_assets_unchanged", "original_model_stats_unchanged", "private_bank_unchanged"):
        require(document.get(key) is True, "source/library/model/bank changed")
    require(type(document["native_sources_verified"]) is int and document["native_sources_verified"] == document["native_sources_verified_after"] == 648,
            "complete fixed native source inventory required")
    native = document["native"]
    require(native.get("schema") == "tc-qwen-ordinary-block-shared-v1" and native.get("status") == "measured" and native.get("errors") == []
            and native.get("case_id") == document["case_id"] and native.get("seed") == document["seed"]
            and native.get("prompt_sha256") == document["prompt_sha256"] and native.get("mlx_version") == "0.32.0", "native request/version binding differs")
    for value in (document, native):
        require(value.get("production_qualified") is False and value.get("physical_overlap_qualified") is False
                and value.get("runtime_internal_copy_behavior") == "unknown", "component evidence cannot grant product/physical/internal-copy qualification")
    require(native.get("shape_mhf") == [1024,4096,12288] and all(type(v) is int for v in native["shape_mhf"])
            and type(native.get("share")) is int and native["share"] == 2048 and type(native.get("source_step")) is int and native["source_step"] == 1
            and type(native.get("source_layer")) is int and native["source_layer"] == 0
            and native.get("block_gate") == {"relative_l2_max": .001, "cosine_min": .999}
            and native.get("ffn_gate") == {"relative_l2_max": .03, "cosine_min": .999}, "fixed full workload and budgets required")
    metric(native["staged_gpu_replay_gate"], .001)
    require(native["staged_gpu_replay_gate"]["max_abs"] == 0, "staged ablation must match original actual block exactly")
    metric(native["independent_full_reference_gate"], .03)
    controls = native["runtime_argument_controls"]
    for key in ("zero_ffn_yields_original_residual", "zero_private_yields_original_tail"):
        metric(controls[key], .001)
        require(controls[key]["max_abs"] == 0, "compiled runtime argument control failed")
    require(type(controls["nonzero_full_ffn_changes_block_elements"]) is int and 0 < controls["nonzero_full_ffn_changes_block_elements"] <= ELEMENTS
            and controls["used_for_speed_ratio"] is False, "FFN must not be omitted from denominator")
    require(len(native["ordinary_argument_sha256"]) == 9 and all(is_sha(v) for v in native["ordinary_argument_sha256"])
            and is_sha(native["ordinary_block_output_sha256"]), "complete original ordinary context identities required")
    require(native.get("private_unloaded") is True and type(native.get("restore_adoption_releases")) is int and native["restore_adoption_releases"] == 2
            and native.get("owned_mapping_release_pass") is True and native.get("borrowed_inputs_outputs_unchanged") is True
            and native.get("mapping_quarantined_until_child_exit", False) is False and native.get("loaded_runtime_id") == document["runtime_build_id"], "normal completed original owner lifecycle required")
    port(native["input_descriptor"])
    port(native["output_descriptor"])
    identity(native["initial_identity"])
    identity(native["final_identity"])
    require(native.get("compile_selector_called") is False and native.get("compile_wall_ns") is None
            and type(native.get("load_wall_ns")) is int and native["load_wall_ns"] > 0, "load is not zero setup or observed compile call")
    require(all(type(native.get(k)) is int and native[k] == value for k, value in (("completed_full_dit_steps",1),("stopped_step",1),("stopped_layer",0)))
            and native.get("vae_decode") is False and native.get("image_export") is False, "block replay is not a completed candidate request/image")
    require(native.get("raw_ffn_artifacts_written") is True and set(artifacts) == {"input", "reference"}
            and set(document["raw_artifacts"]) == {"ffn-input.safetensors", "ffn-reference.safetensors"}, "fresh complete raw pair, not digest-only or archive/reuse required")
    for key, native_key in (("input", "exposed_ffn_input_sha256"), ("reference", "full_ffn_reference_sha256")):
        value = artifacts[key]
        require(value.get("shape") == [1,1024,4096] and value.get("dtype") == "BF16" and value.get("recipe") == RECIPE
                and value.get("payload_bytes") == PAYLOAD_BYTES and value.get("all_finite") is True
                and is_sha(value.get("payload_sha256")) and value["payload_sha256"] == native[native_key]
                and document["raw_artifacts"]["ffn-" + key + ".safetensors"] == {"bytes": value["file_bytes"], "sha256": value["file_sha256"]}, "raw header/full payload/native source identities differ")
    repeats, samples = native["measured_groups"], native["samples"]
    require([(s["phase"],s["trial"],s["arm"]) for s in samples] == expected_order(repeats), "all balanced first/warm/measured block calls required")
    joins, intersections = 0, []
    for sample in samples:
        arm = sample["arm"]
        original, staged, copied = arm == "gpu-original", arm == "gpu-staged", arm == "copy-hybrid"
        private = not original and not staged
        require(type(sample["trial"]) is int and type(sample["wall_ns"]) is int and sample["wall_ns"] > 0
                and sample["output_dtype"] == "bfloat16" and sample["numeric_pass"] is True and sample["guards_pass"] is True
                and sample["physical_overlap_qualified"] is False, "typed complete successful numerical/guard record required")
        expected = {"whole_ordinary_block"} | (set() if original else {"full_ffn", "independent_full_ffn"}) | ({"prefix", "independent_prefix"} if private else set())
        require(set(sample["metrics"]) == expected, "whole block AND full FFN/prefix/independent gates required")
        for key, value in sample["metrics"].items():
            metric(value, .001 if key == "whole_ordinary_block" else .03)
        require(type(sample["external_feature_copy_count"]) is int and sample["external_feature_copy_count"] == (2 if copied else 0)
                and type(sample["external_feature_copy_bytes"]) is int and sample["external_feature_copy_bytes"] == (16777216 if copied else 0), "same-client feature copy accounting differs")
        if original:
            require(sample["metrics"]["whole_ordinary_block"]["max_abs"] == sample["metrics"]["whole_ordinary_block"]["relative_l2"] == 0
                    and sample["host_events"] == [], "actual original callable is the exact complete denominator")
            continue
        names = {"attention-norm-producer-ready", "modulation-residual-output-ready"} | ({"full-gpu-ffn-ready"} if staged else
            {"gpu-pack-ready", "gpu-remainder-ready", "private-evaluate-returned", "gpu-ffn-restore-join-ready"})
        if copied:
            names |= {"cpu-input-feature-copy", "cpu-output-feature-copy"}
        require(len(sample["host_events"]) == len(names) and {e["name"] for e in sample["host_events"]} == names, "complete producer/consumer interval set required")
        table, last = {}, -1
        for event in sample["host_events"]:
            start, duration = event["start_ns"], event["duration_ns"]
            require(type(start) is int and type(duration) is int and 0 <= start and start >= last and duration > 0
                    and start + duration <= sample["wall_ns"] and event["clock_domain"] == "same-process-uptime-raw-host", "associated host-only interval required")
            table[event["name"]], last = (start, start + duration), start
        producer, finish = (table[key] for key in ("attention-norm-producer-ready", "modulation-residual-output-ready"))
        if staged:
            require(producer[1] <= table["full-gpu-ffn-ready"][0] and table["full-gpu-ffn-ready"][1] <= finish[0], "staged dependency differs")
            continue
        pack, ane, gpu, join = (table[key] for key in ("gpu-pack-ready", "private-evaluate-returned", "gpu-remainder-ready", "gpu-ffn-restore-join-ready"))
        supplied = table["cpu-input-feature-copy"][1] if copied else pack[1]
        require(producer[1] <= pack[0] and supplied <= min(ane[0],gpu[0]) and max(ane[1],gpu[1]) <= join[0] and join[1] <= finish[0]
                and (arm != "shared-serial" or ane[1] <= gpu[0]), "complete serial/private/join/downstream dependency differs")
        if copied:
            require(pack[1] <= table["cpu-input-feature-copy"][0] and ane[1] <= table["cpu-output-feature-copy"][0]
                    and table["cpu-output-feature-copy"][1] <= join[0], "feature copy bypassed producer/evaluate")
        identity(sample["identity_before"])
        identity(sample["identity_after"])
        exact = sample["exact_runtime_join"]
        require(type(exact["elements"]) is int and exact["elements"] == ELEMENTS and type(exact["exact_mismatches"]) is int and exact["exact_mismatches"] == 0
                and is_sha(exact["actual_bf16_sha256"]) and exact["actual_bf16_sha256"] == exact["expected_bf16_sha256"], "complete actual-part bitwise join required")
        joins += 1
        intersections.append({"arm": arm, "phase": sample["phase"], "trial": sample["trial"],
                              "api_host_intersection_ns": max(0, min(ane[1],gpu[1])-max(ane[0],gpu[0])), "physical_overlap_qualified": False})
    computed = comparisons(samples, repeats)
    require(document["comparisons"] == computed, "independent fixed-pair arithmetic differs")
    return {"complete_block_calls": len(samples), "exact_component_joins": joins,
            "median_measured_host_wall_ns": {arm: statistics.median(s["wall_ns"] for s in samples if s["phase"] == "measured" and s["arm"] == arm) for arm in ARMS},
            "component_comparisons": computed, "host_intersections": intersections,
            "raw_pair_locally_retained": True, "all_nine_block_arguments_persisted": False,
            "whole_request_qualified": False, "production_qualified": False, "physical_overlap_qualified": False}
