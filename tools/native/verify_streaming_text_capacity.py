"""Verify original Z-Image text-boundary artifacts; never approve a release.

Upper-bound P2 calibration remains required separately. This additional bundle
checks the minimum and encoder/caption boundaries using actual request shapes,
public/manual image parity, native receipts and full process-tree memory logs.
"""
from __future__ import annotations

import copy
import hashlib
import struct
import zlib
from pathlib import Path

from build_streaming_catalog import (
    CanonicalEncoder, catalog_binding, canonical_record_digest,
    validate_record_shape, validate_text_capacity,
)
from generate_runtime_build_identity import seal
from run_streaming_campaign import request_semantic_identity
from verify_process_tree_samples import verify as verify_memory
from verify_streaming_acceptance import Inventory, exact, digest_text
from verify_streaming_campaign import source_provenance_complete
from streaming_release_policy import canonical, CALIBRATED

SCHEMA = "tc-z-image-text-capacity-evidence-v1"
ARTIFACTS = (
    "public_plan", "public_result", "public_resolution", "public_source",
    "public_observation", "public_image", "manual_plan", "manual_result",
    "manual_source", "manual_image", "memory", "memory_summary",
)


def require(condition, message):
    if not condition:
        raise ValueError("text capacity evidence: " + message)


def verify_png(path, width, height):
    with path.open("rb") as stream:
        header = stream.read(33)
        stream.seek(-12, 2)
        end = stream.read(12)
    require(len(header) == 33 and header[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' and
            struct.unpack('>II', header[16:24]) == (width, height) and
            zlib.crc32(header[12:29]) == struct.unpack('>I', header[29:33])[0] and
            end == b'\x00\x00\x00\x00IEND\xaeB`\x82', "PNG dimensions or framing differ")


def required_rows(capacity):
    low, high = capacity["minimum_rows"], capacity["maximum_rows"]
    require(type(low) is int and type(high) is int and 1 <= low <= high <= 1024,
            "invalid token interval")
    return sorted({low, high} | {
        n for center in (32, 512, 1024) for n in (center - 1, center, center + 1)
        if low <= n <= high
    })


def workload_digest(workload):
    out = CanonicalEncoder("tc-streaming-workload-identity-v1")
    for key in ("model", "operation", "execution", "device_class", "execution_container"):
        out.string_field(key, workload[key])
    for key in ("width", "height", "frames", "fps", "steps", "batch"):
        out.unsigned_field(key, workload[key])
    for key in ("audio", "dynamic_text", "approximation"):
        out.boolean_field(key, workload[key])
    for key in ("conditioning_revision", "vae_policy_revision", "feature_digest"):
        out.string_field(key, workload[key])
    out.begin_list("token_shapes", len(workload["token_shapes"]))
    for token in workload["token_shapes"]:
        for key in ("encoder", "tokenizer_revision", "template_revision"):
            out.string_field("token." + key, token[key])
        for key in ("valid_rows", "padded_rows", "compute_rows"):
            out.unsigned_field("token." + key, token[key])
    return out.digest()


def identity_digest(value, kind):
    out = CanonicalEncoder(f"tc-streaming-{kind}-identity-v{2 if kind == 'source' else 1}")
    keys = (("model_variant", "weight_format", "artifact_manifest_digest") if kind == "source" else
            ("turbocider_build_id", "runtime_revision", "adapter_revision", "reader_revision",
             "kernel_revision", "allocator_policy_revision"))
    for key in keys:
        out.string_field(key, value[key])
    return out.digest()


def validate_case(record, rows, documents, *, library_sha256, target, memory_limit):
    """Validate case semantics after Inventory has authenticated every file hash."""
    capacity = record["text_capacity"]
    require(type(rows) is int and capacity["minimum_rows"] <= rows <= capacity["maximum_rows"],
            "case outside token interval")
    workload = copy.deepcopy(record["workload"])
    for key in ("valid_rows", "padded_rows", "compute_rows"):
        workload["token_shapes"][0][key] = rows
    public_plan, manual_plan = documents["public_plan"], documents["manual_plan"]
    left, right = public_plan["request"], manual_plan["request"]
    semantic = request_semantic_identity(left)
    require(semantic == request_semantic_identity(right), "public/manual requests differ")
    def comparison_request(value):
        result = copy.deepcopy(value)
        result["execution"].pop("streaming", None)
        for output in result.get("outputs", []):
            output.pop("path", None)
        return result
    require(comparison_request(left) == comparison_request(right), "public/manual request parameters differ")
    require(left.get("schema_version") == right.get("schema_version") == 2,
            "only V2 image requests are supported")
    for key in ("model", "operation", "execution", "width", "height", "frames", "steps", "audio", "dynamic_text"):
        require(semantic[key] == workload[key], "request differs from record: " + key)
    require(not semantic["compile_gpu"] and not semantic["allow_approximation"] and
            not semantic["nontext_inputs"] and len(semantic["prompts"]) == 1 and
            bool(semantic["prompts"][0]) and not left.get("loras") and not right.get("loras"),
            "unsupported request features")
    require(right["execution"]["streaming"] == record["plan"]["canonical_config"],
            "manual configuration differs from calibrated plan")
    selector = left["execution"]["streaming"]
    require(selector.get("enabled") is True and selector.get("schema_version") == 2 and
            selector.get("target_request_memory_bytes") == target, "public selector differs")
    for route, plan in (("public", public_plan), ("manual", manual_plan)):
        require(plan["library_sha256"] == library_sha256, "case library identity differs")
        config = plan["config"]
        require(config.get("verify_streaming_sources") is True and
                config.get("constructor") == ("public" if route == "public" else "candidate"),
                "wrong constructor or unverified sources")
        if route == "public":
            require(config.get("execution_container") == "cli_worker", "public request is not a worker")
        source = documents[route + "_source"]["report"]
        require(source.get("status") == "verified" and source.get("artifact_manifest_digest") ==
                record["source"]["artifact_manifest_digest"], "source content identity differs")
        result = documents[route + "_result"]
        require(result.get("status") == 0 and not result.get("error"), "generation failed")
        result = result["result"]
        for key in ("model", "operation", "width", "height", "steps"):
            require(result[key] == workload[key], "generated workload differs: " + key)
        require(result["valid_text_tokens"] == rows and result["text_tokens"] == rows and
                result["actual_denoise_steps"] == workload["steps"] and
                result["seed"] == semantic["seed"], "truncated text or incomplete generation")
    public = documents["public_result"]["result"]
    manual = documents["manual_result"]["result"]
    resolved = documents["public_resolution"]
    require(resolved.get("status") == 0 and not resolved.get("error"), "resolution failed")
    resolved = resolved["result"]
    selected = public["public_streaming"]
    layout = selected["actual_layout_digest"]
    require(digest_text(layout) and selected["actual_plan_verified"] is True and
            selected["authorized_layout_digest"] == layout == resolved["selection"]["layout_digest"],
            "actual layout not verified")
    require(selected["record_digest"] == resolved["selection"]["record_digest"] ==
            record["canonical_record_digest"], "selected record differs")
    require(selected["workload_digest"] == resolved["request_digest"] == workload_digest(workload),
            "actual token workload not bound")
    require(selected["resolution_digest"] == resolved["resolution_digest"] ==
            resolved["exact_selector"]["expected_resolution_digest"], "resolution receipt differs")
    for kind in ("source", "runtime"):
        require(selected[kind + "_digest"] == resolved["identity"][kind + "_digest"] ==
                identity_digest(record[kind], kind), kind + " identity differs")
    require(selected["device_digest"] == resolved["identity"]["device_digest"] and
            digest_text(selected["device_digest"]), "device identity differs")
    require(selected["execution_container"] == "cli_worker" and
            selected["memory_scope"] == "execution_process_tree_v1" and
            selected["target_request_memory_bytes"] == target and
            selected["component_policy_revision"] == record["plan"]["component_policy_revision"],
            "execution or memory scope differs")
    if rows == capacity["maximum_rows"]:
        require(layout == record["plan"]["layout_digest"], "upper-bound layout differs")
    require(len(public["streaming_stages"]) == 1, "unexpected stage count")
    public_runtime = public["streaming_stages"][0]["runtime"]
    require(public_runtime["digest"] == layout and public_runtime["source_lease_verified"] is True,
            "public stage not source/layout verified")
    # The manual reference uses the same weights/math/slot plan but retains its
    # pool through VAE. Its private-layout digest and lease receipt differ by
    # design; only the public path supplies execution authority and receipts.
    manual_runtime = manual["block_streaming"]["actual_layout"]
    for runtime in (public_runtime, manual_runtime):
        require(runtime["drained"] is True and runtime["pass_count"] == workload["steps"] and
                runtime["kernel_revision"] == record["runtime"]["kernel_revision"] and
                runtime["weight_format"] == record["source"]["weight_format"], "stage not completed")
        for key in ("resident_prefix_blocks", "block_group_size", "slot_count", "prefetch_distance", "io_workers"):
            require(runtime[key] == record["plan"]["canonical_config"]["stages"]["denoiser"][key],
                    "stage policy differs")
    receipt = public["streaming_stages"][0]["runtime"]["receipt"]
    require(digest_text(receipt["canonical_digest"]) and receipt["canonical_digest"] == selected["receipt_digest"] and
            receipt["schema_version"] == selected["receipt_schema_version"] == 2 and
            receipt["source_generation"] == selected["receipt_source_generation"] > 0 and
            receipt["reader_fences_issued"] == receipt["reader_fences_completed"] > 0,
            "incomplete native receipt")
    observed = documents["public_observation"]
    require(observed.get("status") == "succeeded" and observed.get("cleanup_returned") is True and
            observed.get("cancellation_sent") is False, "request did not finish and release")
    memory = documents["memory_summary"]
    require(observed.get("process_identity") == memory.get("root_identity") and
            isinstance(observed.get("process_identity"), dict), "memory sampled a different process")
    require(memory.get("complete") is True and memory.get("command_exit_code") == 0 and
            0 < memory["tree_peak_phys_footprint_bytes"] <= memory_limit and
            0 < memory["max_gap_ns"] <= memory["allowed_max_gap_ns"] <= 100_000_000 and
            memory["swap_out_bytes"] == 0, "incomplete or out-of-budget memory evidence")
    return dict(rows=rows, workload_digest=selected["workload_digest"], layout_digest=layout,
                device_digest=selected["device_digest"], peak_bytes=memory["tree_peak_phys_footprint_bytes"],
                maximum_sample_gap_ns=memory["max_gap_ns"],
                swap_in_bytes=memory["swap_in_bytes"], swap_out_bytes=memory["swap_out_bytes"])


def verify(bundle: Path, binding: dict, reviewed_commit: str):
    inventory = Inventory(bundle)
    try:
        raw, range_file = inventory.read("range.json", json_limit=True)
        evidence = inventory.decode(raw)
        exact(evidence, ("schema", "binding", "reviewed_commit", "source_identity", "build_manifest",
                         "library_sha256", "catalog", "target_bytes", "cases"), "text range")
        require(evidence["schema"] == SCHEMA and evidence["binding"] == binding and
                evidence["reviewed_commit"] == reviewed_commit, "range binding/commit differs")
        require(binding.get("release_policy_revision") == CALIBRATED, "requires calibrated release policy")
        provenance = inventory.referenced(evidence["source_identity"], json_limit=True)
        require(source_provenance_complete({"source_identity": provenance}) and provenance.get("clean") is True and
                provenance["commit"] == reviewed_commit,
                "missing clean source provenance")
        build = inventory.referenced(evidence["build_manifest"], json_limit=True)
        require(build == seal(build["inputs"]) and
                build["catalog_runtime_id"] == binding["runtime"]["turbocider_build_id"],
                "native manifest identity differs")
        require(digest_text(evidence["library_sha256"]), "missing library hash")
        build_sources = {entry['path']: entry['sha256'] for entry in build['inputs']['sources']}
        catalog = inventory.referenced(evidence["catalog"], json_limit=True)
        require(len(catalog["records"]) == 1, "expected one immutable capacity record")
        record = catalog["records"][0]
        validate_record_shape(record, allow_test_template=True)
        validate_text_capacity(record)
        require("text_capacity" in record and catalog_binding(record) == binding and
                canonical_record_digest(record) == record["canonical_record_digest"], "catalog binding differs")
        target = evidence["target_bytes"]
        require(type(target) is int and target in (n << 30 for n in (8, 10, 12, 16, 20)), "unsupported target")
        memory_limit = target - max(512 << 20, (target + 9) // 10)
        cases = evidence["cases"]
        require(isinstance(cases, list) and 1 <= len(cases) <= 64, "invalid boundary cases")
        seen, checked = set(), []
        memory_roots = set()
        for case in cases:
            exact(case, ("rows", *ARTIFACTS), "boundary case")
            rows = case["rows"]
            require(type(rows) is int and rows not in seen, "duplicate/invalid token length")
            seen.add(rows)
            documents = {}
            for key in ARTIFACTS:
                documents[key] = inventory.referenced(case[key], json_limit=key not in ("public_image", "manual_image", "memory"))
            require(case["public_image"]["sha256"] == case["manual_image"]["sha256"] ==
                    documents["public_observation"]["image_sha256"], "public/manual PNGs differ")
            for route in ('public', 'manual'):
                artifact = case[route + '_image']
                verify_png(bundle / artifact['path'], binding['workload']['width'], binding['workload']['height'])
                inventory.referenced(artifact)
                script = 'run_public_streaming_smoke.py' if route == 'public' else 'run_image_streaming_smoke.py'
                require(documents[route + '_plan']['script_sha256'] == build_sources['tools/native/' + script],
                        'case producer differs from measured build inputs')
            memory = verify_memory(bundle / case["memory"]["path"])
            inventory.referenced(case["memory"])  # Reject changes during independent parsing.
            reported = documents["memory_summary"]
            root_identity = canonical(reported["root_identity"])
            require(root_identity not in memory_roots, "reused process memory for multiple boundaries")
            memory_roots.add(root_identity)
            for key in ("complete", "command_exit_code", "tree_peak_phys_footprint_bytes", "max_gap_ns",
                        "allowed_max_gap_ns", "swap_in_bytes", "swap_out_bytes", "correlation_id", "root_identity"):
                require(memory[key] == reported[key], "memory summary differs from raw log: " + key)
            require(memory["final_evidence_digest"] == reported["evidence_digest"], "memory log digest differs")
            require(documents["public_plan"]["catalog_sha256"] == evidence["catalog"]["sha256"], "case catalog hash differs")
            checked.append(validate_case(record, rows, documents, library_sha256=evidence["library_sha256"],
                                         target=target, memory_limit=memory_limit))
        require(set(required_rows(record["text_capacity"])).issubset(seen), "missing required text boundaries")
        require(len({c["device_digest"] for c in checked}) == 1, "device changed between cases")
        return dict(schema="tc-z-image-text-capacity-verification-v1", passed=True,
                    reviewed_commit=reviewed_commit, binding_sha256=hashlib.sha256(canonical(binding)).hexdigest(),
                    target_bytes=target, range_file=range_file,
                    maximum_peak_bytes=max(c["peak_bytes"] for c in checked),
                    maximum_sample_gap_ns=max(c["maximum_sample_gap_ns"] for c in checked),
                    checks=checked, artifacts=list(inventory.verified.values()), production_authorized=False)
    finally:
        inventory.close()
