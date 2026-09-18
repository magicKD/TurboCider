#!/usr/bin/env python3
"""Build a deterministic streaming preset record from verified evidence.

This tool is intentionally conservative.  It creates a reviewable record
artifact; it never edits the production C++ catalog and it never upgrades an
inconclusive campaign.  The input record contains model/runtime identities and
the exact plan observed by the model adapter, while the campaign bundle proves
the request, performance, quality and process-tree memory gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from verify_streaming_campaign import EvidenceError, verify as verify_campaign


SCHEMA = "turbocider-streaming-catalog-build-v1"
BUILDER_REVISION = "tc-streaming-catalog-builder-v1"
PUBLIC_TARGETS = {value << 30 for value in (8, 10, 12, 16, 20)}
HEADROOM_REVISION = "tc-public-headroom-v1"
RECORD_SCHEMA = "tc-streaming-preset-record-v1"
RECORD_DIGEST_SCHEMA = "tc-streaming-preset-record-digest-v1"
RECORD_IDENTITY_SCHEMA = "tc-streaming-preset-record-identity-v1"
REVIEW_SCHEMA = "tc-streaming-catalog-review-v1"
CALIBRATION_SCOPE = "execution_process_tree_v1"
CALIBRATION_ESTIMATOR = "tree-phys-footprint-linear-p95-v1"
STAGING_GATES = ("P0", "P1", "P2")
PUBLIC_GATES = (*STAGING_GATES, "P3")
REVIEW_ROLES = ("runtime", "model", "performance", "release")


class CatalogBuildError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise CatalogBuildError(f"cannot read {label}: {exc}") from exc
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CatalogBuildError(
                f"cannot parse {label} line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CatalogBuildError(
                f"{label} line {line_number} must contain an object"
            )
        values.append(value)
    return values


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogBuildError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogBuildError(f"{label} must be a JSON object")
    return value


def require_string(value: Any, label: str, *, digest: bool = False) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise CatalogBuildError(f"{label} must be a non-empty string")
    if digest and (
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise CatalogBuildError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_commit(value: Any, label: str) -> str:
    value = require_string(value, label)
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise CatalogBuildError(f"{label} must be a lowercase 40-character commit")
    return value


def require_uint(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogBuildError(f"{label} must be an unsigned integer")
    if positive and value == 0:
        raise CatalogBuildError(f"{label} must be positive")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CatalogBuildError(f"{label} must be boolean")
    return value


def require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise CatalogBuildError(f"{label} is missing: {', '.join(missing)}")
    if unknown:
        raise CatalogBuildError(f"{label} has unencoded fields: {', '.join(unknown)}")


def validate_config_shape(config: dict[str, Any]) -> None:
    require_exact_keys(
        config,
        {"enabled", "schema_version", "selection", "retention", "stages"},
        set(),
        "record.plan.canonical_config",
    )
    if config["enabled"] is not True:
        raise CatalogBuildError("record.plan.canonical_config must be active")
    if config["schema_version"] != 1:
        raise CatalogBuildError("record.plan.canonical_config.schema_version must be 1")
    if config["selection"] != "manual" or config["retention"] != "request":
        raise CatalogBuildError(
            "record.plan.canonical_config must use manual/request"
        )
    stages = config["stages"]
    if not isinstance(stages, dict) or not stages or len(stages) > 64:
        raise CatalogBuildError(
            "record.plan.canonical_config.stages must contain 1...64 stages"
        )
    fields = {
        "residency", "block_group_size", "slot_count",
        "resident_prefix_blocks", "prefetch_distance", "io_workers",
    }
    for stage_id, stage in stages.items():
        require_string(stage_id, "record.plan stage id")
        if len(stage_id) > 128 or not isinstance(stage, dict):
            raise CatalogBuildError(f"record.plan stage {stage_id!r} is invalid")
        residency = stage.get("residency")
        if residency == "resident":
            require_exact_keys(
                stage, {"residency"}, set(),
                f"record.plan.canonical_config.stages.{stage_id}",
            )
            continue
        if residency != "streamed":
            raise CatalogBuildError(f"record.plan stage {stage_id} has invalid residency")
        require_exact_keys(
            stage, fields, set(),
            f"record.plan.canonical_config.stages.{stage_id}",
        )
        group = require_uint(
            stage["block_group_size"], f"record.plan stage {stage_id}.block_group_size",
            positive=True,
        )
        slots = require_uint(
            stage["slot_count"], f"record.plan stage {stage_id}.slot_count",
            positive=True,
        )
        require_uint(
            stage["resident_prefix_blocks"],
            f"record.plan stage {stage_id}.resident_prefix_blocks",
        )
        distance = require_uint(
            stage["prefetch_distance"],
            f"record.plan stage {stage_id}.prefetch_distance",
        )
        workers = require_uint(
            stage["io_workers"], f"record.plan stage {stage_id}.io_workers",
            positive=True,
        )
        if group == 0 or distance >= slots or workers > slots:
            raise CatalogBuildError(f"record.plan stage {stage_id} slot policy is invalid")


def validate_record_shape(record: dict[str, Any]) -> None:
    require_exact_keys(
        record,
        {
            "id", "revision", "catalog_revision", "source", "workload",
            "runtime", "device", "plan", "calibration", "performance",
            "release",
        },
        {"canonical_record_digest"},
        "record",
    )
    for key in (
        "id", "catalog_revision", "source", "workload", "runtime",
        "device", "plan", "calibration", "performance", "release",
    ):
        if key not in record:
            raise CatalogBuildError(f"record is missing {key}")
    require_string(record["id"], "record.id")
    require_string(record["catalog_revision"], "record.catalog_revision")
    revision = require_uint(record.get("revision"), "record.revision", positive=True)
    if revision <= 0:
        raise CatalogBuildError("record.revision must be positive")
    for section in ("source", "workload", "runtime", "device", "plan", "calibration", "performance", "release"):
        if not isinstance(record[section], dict):
            raise CatalogBuildError(f"record.{section} must be an object")
    source = record["source"]
    require_exact_keys(
        source,
        {
            "model_variant", "weight_format", "artifact_manifest_digest",
            "source_snapshot_digest",
        },
        set(),
        "record.source",
    )
    for key in ("model_variant", "weight_format"):
        require_string(source.get(key), f"record.source.{key}")
    for key in ("artifact_manifest_digest", "source_snapshot_digest"):
        require_string(source.get(key), f"record.source.{key}", digest=True)
    workload = record["workload"]
    require_exact_keys(
        workload,
        {
            "model", "operation", "execution", "device_class",
            "execution_container", "width", "height", "frames", "fps",
            "steps", "batch", "audio", "dynamic_text", "approximation",
            "conditioning_revision", "vae_policy_revision", "feature_digest",
        },
        {"token_shapes"},
        "record.workload",
    )
    for key in (
        "model", "operation", "execution", "device_class",
        "execution_container", "conditioning_revision", "vae_policy_revision",
        "feature_digest",
    ):
        require_string(
            workload.get(key), f"record.workload.{key}",
            digest=key == "feature_digest",
        )
    for key in ("width", "height", "frames", "fps", "steps", "batch"):
        require_uint(workload.get(key), f"record.workload.{key}", positive=key != "fps")
    require_bool(workload.get("audio"), "record.workload.audio")
    require_bool(workload.get("dynamic_text"), "record.workload.dynamic_text")
    require_bool(workload.get("approximation"), "record.workload.approximation")
    token_shapes = workload.get("token_shapes", [])
    if not isinstance(token_shapes, list) or len(token_shapes) > 8:
        raise CatalogBuildError("record.workload.token_shapes must contain at most 8 items")
    for index, token in enumerate(token_shapes):
        if not isinstance(token, dict):
            raise CatalogBuildError(f"record.workload.token_shapes[{index}] must be an object")
        for key in ("encoder", "tokenizer_revision", "template_revision"):
            require_string(token.get(key), f"record.workload.token_shapes[{index}].{key}")
        valid_rows = require_uint(
            token.get("valid_rows"),
            f"record.workload.token_shapes[{index}].valid_rows",
            positive=True,
        )
        padded_rows = require_uint(
            token.get("padded_rows"),
            f"record.workload.token_shapes[{index}].padded_rows",
            positive=True,
        )
        compute_rows = require_uint(
            token.get("compute_rows"),
            f"record.workload.token_shapes[{index}].compute_rows",
            positive=True,
        )
        if valid_rows > padded_rows or padded_rows > compute_rows:
            raise CatalogBuildError(
                f"record.workload.token_shapes[{index}] rows are not monotonic"
            )
    runtime = record["runtime"]
    require_exact_keys(
        runtime,
        {
            "turbocider_build_id", "runtime_revision", "adapter_revision",
            "reader_revision", "kernel_revision", "allocator_policy_revision",
        },
        set(),
        "record.runtime",
    )
    for key in (
        "turbocider_build_id", "runtime_revision", "adapter_revision",
        "reader_revision", "kernel_revision", "allocator_policy_revision",
    ):
        require_string(runtime.get(key), f"record.runtime.{key}")
    device = record["device"]
    require_exact_keys(
        device,
        {"minimum_physical_memory_bytes", "maximum_physical_memory_bytes"},
        set(),
        "record.device",
    )
    minimum = require_uint(
        device.get("minimum_physical_memory_bytes"),
        "record.device.minimum_physical_memory_bytes", positive=True,
    )
    maximum = require_uint(
        device.get("maximum_physical_memory_bytes", 0),
        "record.device.maximum_physical_memory_bytes",
    )
    if maximum and maximum < minimum:
        raise CatalogBuildError("record.device memory range is inverted")
    plan = record["plan"]
    require_exact_keys(
        plan,
        {
            "canonical_config", "layout_digest", "component_policy_revision",
            "pass_transition", "multi_pool_policy",
        },
        set(),
        "record.plan",
    )
    for key in (
        "canonical_config", "layout_digest", "component_policy_revision",
        "pass_transition", "multi_pool_policy",
    ):
        if key not in plan:
            raise CatalogBuildError(f"record.plan is missing {key}")
    require_string(plan["layout_digest"], "record.plan.layout_digest", digest=True)
    require_string(plan["component_policy_revision"], "record.plan.component_policy_revision")
    require_string(plan["pass_transition"], "record.plan.pass_transition")
    require_string(plan["multi_pool_policy"], "record.plan.multi_pool_policy")
    if not isinstance(plan["canonical_config"], dict):
        raise CatalogBuildError("record.plan.canonical_config must be an object")
    validate_config_shape(plan["canonical_config"])
    calibration = record["calibration"]
    require_exact_keys(
        calibration,
        {
            "complete", "calibrated_request_bytes", "scope",
            "estimator_revision", "calibration_id", "execution_container",
            "evidence_digest", "confirmation_sample_count",
            "maximum_sample_gap_ns",
        },
        set(),
        "record.calibration",
    )
    if calibration.get("complete") is not True:
        raise CatalogBuildError("record.calibration.complete must be true")
    for key in ("calibrated_request_bytes", "confirmation_sample_count", "maximum_sample_gap_ns"):
        require_uint(calibration.get(key), f"record.calibration.{key}", positive=key != "maximum_sample_gap_ns")
    for key in ("scope", "estimator_revision", "calibration_id", "execution_container"):
        require_string(calibration.get(key), f"record.calibration.{key}")
    if calibration["scope"] != CALIBRATION_SCOPE:
        raise CatalogBuildError("record.calibration.scope is unsupported")
    if calibration["estimator_revision"] != CALIBRATION_ESTIMATOR:
        raise CatalogBuildError("record.calibration.estimator_revision is unsupported")
    if calibration["execution_container"] != workload["execution_container"]:
        raise CatalogBuildError("record calibration/workload container differs")
    require_string(calibration.get("evidence_digest"), "record.calibration.evidence_digest", digest=True)
    performance = record["performance"]
    require_exact_keys(
        performance,
        {
            "rank", "logical_read_bytes", "profile_id", "comparison_kind",
            "confidence_status", "evidence_digest",
        },
        set(),
        "record.performance",
    )
    require_uint(performance.get("rank"), "record.performance.rank")
    require_uint(performance.get("logical_read_bytes"), "record.performance.logical_read_bytes")
    for key in ("profile_id", "comparison_kind", "confidence_status"):
        require_string(performance.get(key), f"record.performance.{key}")
    require_string(performance.get("evidence_digest"), "record.performance.evidence_digest", digest=True)
    if performance["comparison_kind"] != "P1_same_layout":
        raise CatalogBuildError("record.performance.comparison_kind must be P1_same_layout")
    if performance["confidence_status"] != "PASS":
        raise CatalogBuildError("record.performance.confidence_status must be PASS")
    release = record["release"]
    require_exact_keys(
        release,
        {"channel", "revoked", "reviewed_commit", "review_digest"},
        set(),
        "record.release",
    )
    if release.get("channel") not in ("staging", "public-stable", "public-experimental", "revoked"):
        raise CatalogBuildError("record.release.channel is invalid")
    if release.get("revoked") is not False:
        raise CatalogBuildError("new catalog records cannot be revoked")
    require_string(release.get("reviewed_commit"), "record.release.reviewed_commit")
    require_string(release.get("review_digest"), "record.release.review_digest", digest=True)


class CanonicalEncoder:
    def __init__(self, schema: str):
        self.value = bytearray(b"H")
        self.string(schema)

    def tag(self, value: str) -> None:
        self.value.extend(value.encode("ascii"))

    def u64(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << 64):
            raise CatalogBuildError("canonical unsigned value is outside uint64")
        self.value.extend(int(value).to_bytes(8, "big", signed=False))

    def string(self, value: str | bytes) -> None:
        raw = value.encode() if isinstance(value, str) else bytes(value)
        self.u64(len(raw))
        self.value.extend(raw)

    def field(self, value: str) -> None:
        self.string(value)

    def string_field(self, name: str, value: str | bytes) -> None:
        self.tag("S"); self.field(name); self.string(value)

    def optional_string_field(self, name: str, value: Any) -> None:
        self.tag("O"); self.field(name)
        if value is None:
            self.tag("0")
        else:
            self.tag("1"); self.string(value)

    def unsigned_field(self, name: str, value: int) -> None:
        self.tag("U"); self.field(name); self.u64(value)

    def boolean_field(self, name: str, value: bool) -> None:
        self.tag("B"); self.field(name); self.tag("1" if value else "0")

    def begin_list(self, name: str, count: int) -> None:
        self.tag("L"); self.field(name); self.u64(count)

    def digest(self) -> str:
        return sha256_bytes(bytes(self.value))

    def bytes(self) -> bytes:
        return bytes(self.value)


def encode_config(out: CanonicalEncoder, config: dict[str, Any]) -> None:
    enabled = config.get("enabled")
    out.boolean_field("config.enabled.present", enabled is not None)
    if enabled is not None:
        out.boolean_field("config.enabled.value", enabled)
    version = config.get("schema_version")
    out.boolean_field("config.schema_version.present", version is not None)
    if version is not None:
        out.unsigned_field("config.schema_version.value", version)
    out.optional_string_field("config.selection", config.get("selection"))
    out.optional_string_field("config.retention", config.get("retention"))
    stages = config.get("stages", {})
    if not isinstance(stages, dict):
        raise CatalogBuildError("plan.canonical_config.stages must be an object")
    out.begin_list("config.stages", len(stages))
    for stage_id in sorted(stages):
        stage = stages[stage_id]
        if not isinstance(stage, dict):
            raise CatalogBuildError(f"stage {stage_id} must be an object")
        out.string_field("stage.id", stage_id)
        out.optional_string_field("stage.residency", stage.get("residency"))
        for key in ("block_group_size", "slot_count", "resident_prefix_blocks", "prefetch_distance", "io_workers"):
            value = stage.get(key)
            out.boolean_field(f"stage.{key}.present", value is not None)
            if value is not None:
                out.unsigned_field(f"stage.{key}.value", value)


def encode_record_fields(
    out: CanonicalEncoder,
    record: dict[str, Any],
    *,
    include_id: bool,
    include_release: bool,
) -> None:
    """Encode exactly the field order used by preset_catalog.cpp.

    The record digest is deliberately not a JSON digest.  The native catalog
    uses a binary length-prefixed encoder and then wraps the canonical bytes
    in a second digest schema.  Keeping the field walk in one function makes
    the identity, canonical-record, and native digest paths auditable.
    """
    if include_id:
        out.string_field("id", record["id"])
    out.unsigned_field("revision", record["revision"])
    out.string_field("catalog_revision", record["catalog_revision"])
    source = record["source"]
    for key in ("model_variant", "weight_format", "artifact_manifest_digest", "source_snapshot_digest"):
        out.string_field(f"source.{key}", source[key])
    workload = record["workload"]
    for key in ("model", "operation", "execution", "device_class", "execution_container"):
        out.string_field(f"workload.{key}", workload[key])
    for key in ("width", "height", "frames", "fps", "steps", "batch"):
        out.unsigned_field(f"workload.{key}", workload[key])
    for key in ("audio", "dynamic_text", "approximation"):
        out.boolean_field(f"workload.{key}", workload[key])
    for key in ("conditioning_revision", "vae_policy_revision", "feature_digest"):
        out.string_field(f"workload.{key}", workload[key])
    tokens = workload.get("token_shapes", [])
    out.begin_list("workload.token_shapes", len(tokens))
    for token in tokens:
        for key in ("encoder", "tokenizer_revision", "template_revision"):
            out.string_field(f"token.{key}", token[key])
        for key in ("valid_rows", "padded_rows", "compute_rows"):
            out.unsigned_field(f"token.{key}", token[key])
    runtime = record["runtime"]
    for key in ("turbocider_build_id", "runtime_revision", "adapter_revision", "reader_revision", "kernel_revision", "allocator_policy_revision"):
        out.string_field(f"runtime.{key}", runtime[key])
    device = record["device"]
    out.unsigned_field("device.minimum_physical_memory_bytes", device["minimum_physical_memory_bytes"])
    out.unsigned_field("device.maximum_physical_memory_bytes", device["maximum_physical_memory_bytes"])
    plan = record["plan"]
    encode_config(out, plan["canonical_config"])
    for key in ("layout_digest", "component_policy_revision", "pass_transition", "multi_pool_policy"):
        out.string_field(f"plan.{key}", plan[key])
    calibration = record["calibration"]
    out.boolean_field("calibration.complete", calibration["complete"])
    out.unsigned_field(
        "calibration.calibrated_request_bytes",
        calibration["calibrated_request_bytes"],
    )
    for key in (
        "scope", "estimator_revision", "calibration_id",
        "execution_container", "evidence_digest",
    ):
        out.string_field(f"calibration.{key}", calibration[key])
    out.unsigned_field(
        "calibration.confirmation_sample_count",
        calibration["confirmation_sample_count"],
    )
    out.unsigned_field(
        "calibration.maximum_sample_gap_ns",
        calibration["maximum_sample_gap_ns"],
    )
    performance = record["performance"]
    out.unsigned_field("performance.rank", performance["rank"])
    out.unsigned_field("performance.logical_read_bytes", performance["logical_read_bytes"])
    for key in ("profile_id", "comparison_kind", "confidence_status", "evidence_digest"):
        out.string_field(f"performance.{key}", performance[key])
    if include_release:
        release = record["release"]
        out.string_field("release.channel", release["channel"])
        out.boolean_field("release.revoked", release["revoked"])
        for key in ("reviewed_commit", "review_digest"):
            out.string_field(f"release.{key}", release[key])


def canonical_record_bytes(record: dict[str, Any]) -> bytes:
    out = CanonicalEncoder(RECORD_SCHEMA)
    encode_record_fields(out, record, include_id=True, include_release=True)
    return out.bytes()


def canonical_record(record: dict[str, Any]) -> bytes:
    """Compatibility alias returning native canonical record bytes."""
    return canonical_record_bytes(record)


def canonical_record_digest(record: dict[str, Any]) -> str:
    out = CanonicalEncoder(RECORD_DIGEST_SCHEMA)
    out.string_field("canonical_record", canonical_record_bytes(record))
    return out.digest()


def record_identity_digest(record: dict[str, Any]) -> str:
    out = CanonicalEncoder(RECORD_IDENTITY_SCHEMA)
    encode_record_fields(out, record, include_id=False, include_release=False)
    return out.digest()


def catalog_binding(record: dict[str, Any]) -> dict[str, Any]:
    performance = record["performance"]
    return {
        "source": copy.deepcopy(record["source"]),
        "workload": copy.deepcopy(record["workload"]),
        "runtime": copy.deepcopy(record["runtime"]),
        "device": copy.deepcopy(record["device"]),
        "plan": copy.deepcopy(record["plan"]),
        "performance_profile": {
            "rank": performance["rank"],
            "logical_read_bytes": performance["logical_read_bytes"],
            "profile_id": performance["profile_id"],
            "comparison_kind": performance["comparison_kind"],
        },
    }


def _verified_summary(
    bundle: Path, expected_kind: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        summary = verify_campaign(bundle)
    except EvidenceError as exc:
        raise CatalogBuildError(f"campaign evidence is invalid: {exc}") from exc
    if summary.get("overall") != "PASS":
        raise CatalogBuildError(
            f"campaign verifier is not PASS: {summary.get('overall')}"
        )
    if summary.get("comparison_kind") != expected_kind:
        raise CatalogBuildError(
            f"expected {expected_kind} evidence, got {summary.get('comparison_kind')}"
        )
    summary_path = bundle / "summary.json"
    persisted = read_object(summary_path, f"{expected_kind} campaign summary")
    if canonical_json(persisted) != canonical_json(summary):
        raise CatalogBuildError(
            f"{expected_kind} persisted summary differs from independent verifier"
        )
    summary_digest = sha256_file(summary_path)
    policy = read_object(bundle / "campaign-policy.json", "campaign policy")
    if policy.get("comparison_kind") != expected_kind:
        raise CatalogBuildError(f"{expected_kind} policy kind differs")
    return summary, policy, summary_digest


def _validate_binding(
    policy: dict[str, Any], record: dict[str, Any], kind: str
) -> None:
    binding = policy.get("catalog_binding")
    if not isinstance(binding, dict):
        raise CatalogBuildError(f"{kind} policy lacks catalog_binding")
    if canonical_json(binding) != canonical_json(catalog_binding(record)):
        raise CatalogBuildError(f"{kind} catalog binding differs from record")


def _validate_commit_identity(
    summary: dict[str, Any], reviewed_commit: str, kind: str, public: bool
) -> None:
    build = summary.get("build_identity")
    candidate = build.get("candidate") if isinstance(build, dict) else None
    source = candidate.get("source_identity") if isinstance(candidate, dict) else None
    if not isinstance(source, dict) or source.get("commit") != reviewed_commit:
        raise CatalogBuildError(
            f"{kind} candidate source commit differs from review"
        )
    if public and source.get("clean") is not True:
        raise CatalogBuildError(
            f"{kind} public evidence must come from a clean worktree"
        )


def _validate_p2_summary(
    bundle: Path,
    summary: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], int, str]:
    memory = summary.get("memory_evidence")
    if not isinstance(memory, dict) or memory.get("qualification") != "PASS":
        raise CatalogBuildError("memory evidence is not a verified PASS")
    for key in (
        "over_target", "unexpected_swap", "incomplete", "insufficient_variants",
        "fresh_process_failures",
    ):
        if memory.get(key):
            raise CatalogBuildError(f"memory evidence has non-empty {key}")
    required_count = memory.get("required_count")
    if (
        isinstance(required_count, bool) or not isinstance(required_count, int) or
        required_count < 20
    ):
        raise CatalogBuildError("P2 memory evidence requires at least 20 runs")
    config = policy.get("memory_sampling")
    if not isinstance(config, dict) or config.get("headroom_policy_revision") != HEADROOM_REVISION:
        raise CatalogBuildError("campaign does not use the reviewed headroom policy")
    target = config.get("target_bytes")
    if target not in PUBLIC_TARGETS:
        raise CatalogBuildError("campaign target is not a public memory tier")
    if memory.get("required_variants") != ["candidate"]:
        raise CatalogBuildError(
            "P2 catalog evidence must measure candidate memory only"
        )
    if memory.get("target_bytes") != target:
        raise CatalogBuildError("memory evidence target differs from frozen policy")
    allowed_peak = memory.get("allowed_peak_bytes")
    if isinstance(allowed_peak, bool) or not isinstance(allowed_peak, int) or allowed_peak <= 0:
        raise CatalogBuildError("memory evidence allowed peak is invalid")
    maximum_gap = memory.get("maximum_sample_gap_ns")
    allowed_gap = memory.get("allowed_max_gap_ns")
    if (
        isinstance(maximum_gap, bool) or not isinstance(maximum_gap, int) or
        isinstance(allowed_gap, bool) or not isinstance(allowed_gap, int) or
        maximum_gap > allowed_gap
    ):
        raise CatalogBuildError("memory evidence contains an excessive sample gap")
    peaks = memory.get("peak_p95_bytes")
    if not isinstance(peaks, dict):
        raise CatalogBuildError("memory evidence lacks peak_p95_bytes")
    candidate_peak = peaks.get("candidate")
    if isinstance(candidate_peak, bool) or not isinstance(candidate_peak, (int, float)):
        raise CatalogBuildError("memory evidence lacks candidate peak P95")
    if not math.isfinite(float(candidate_peak)) or candidate_peak <= 0:
        raise CatalogBuildError("memory evidence candidate peak P95 is invalid")
    return memory, target, sha256_file(bundle / "summary.json")


def _validate_p3_summary(
    summary: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    memory = summary.get("memory_evidence")
    result = summary.get("p3_result")
    if not isinstance(memory, dict) or memory.get("qualification") != "PASS":
        raise CatalogBuildError("P3 memory evidence is not a verified PASS")
    if not isinstance(result, dict) or result.get("qualification") != "PASS":
        raise CatalogBuildError("P3 natural-swap result is not a verified PASS")
    if result.get("classification") not in (
        "faster_and_lower_swap", "lower_swap_tradeoff",
    ):
        raise CatalogBuildError("P3 natural-swap classification is invalid")
    config = policy.get("memory_sampling")
    if not isinstance(config, dict) or config.get("required_variants") != [
        "baseline", "candidate"
    ]:
        raise CatalogBuildError(
            "P3 must measure baseline and candidate process trees"
        )
    if config.get("allow_swap_out") is not True:
        raise CatalogBuildError("P3 policy does not allow observed natural swap")
    counts = memory.get("required_count_by_variant")
    if not isinstance(counts, dict) or any(
        isinstance(counts.get(variant), bool) or
        not isinstance(counts.get(variant), int) or
        counts[variant] < 20
        for variant in ("baseline", "candidate")
    ):
        raise CatalogBuildError(
            "P3 requires at least 20 measured runs per variant"
        )
    totals = memory.get("swap_out_total_bytes")
    if not isinstance(totals, dict) or any(
        isinstance(totals.get(variant), bool) or
        not isinstance(totals.get(variant), int) or totals[variant] < 0
        for variant in ("baseline", "candidate")
    ):
        raise CatalogBuildError("P3 swap-out totals are invalid")
    if totals["baseline"] <= 0:
        raise CatalogBuildError("P3 baseline did not exhibit observable swap-out")
    ratio = memory.get("candidate_swap_out_total_ratio")
    if (
        isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or
        not math.isfinite(float(ratio)) or ratio < 0
    ):
        raise CatalogBuildError("P3 candidate swap-out ratio is invalid")
    comparison = policy.get("swap_comparison")
    maximum = (
        comparison.get("candidate_swap_out_total_ratio_max")
        if isinstance(comparison, dict) else None
    )
    if (
        isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or
        not math.isfinite(float(maximum)) or not 0 <= maximum <= 1
    ):
        raise CatalogBuildError("P3 policy swap-out ratio maximum is invalid")
    if ratio > maximum:
        raise CatalogBuildError("P3 candidate swap-out ratio exceeds policy")
    return result


def _candidate_layout_digest(bundle: Path) -> str:
    rows = read_jsonl(bundle / "raw-samples.jsonl", "campaign raw samples")
    layouts = {
        row.get("actual_layout_digest")
        for row in rows
        if isinstance(row, dict) and row.get("variant") == "candidate" and
        row.get("status") == "success"
    }
    if len(layouts) != 1 or None in layouts:
        raise CatalogBuildError("candidate evidence does not have one layout digest")
    return next(iter(layouts))


def _review_digest(review: dict[str, Any]) -> str:
    payload = copy.deepcopy(review)
    payload.pop("review_digest", None)
    return sha256_bytes(canonical_json(payload))


def _validate_review(
    review: dict[str, Any],
    record: dict[str, Any],
    evidence_digests: dict[str, str],
    identity_digest: str,
    required_gates: tuple[str, ...],
) -> str:
    require_exact_keys(
        review,
        {
            "schema", "status", "reviewed_commit", "record_identity_digest",
            "evidence_summary_sha256", "reviewers", "review_digest",
        },
        set(),
        "review",
    )
    if review["schema"] != REVIEW_SCHEMA or review["status"] != "approved":
        raise CatalogBuildError("review is not an approved v1 review")
    reviewed_commit = require_commit(review["reviewed_commit"], "review.reviewed_commit")
    require_string(
        review["record_identity_digest"],
        "review.record_identity_digest",
        digest=True,
    )
    if review["record_identity_digest"] != identity_digest:
        raise CatalogBuildError("review record identity differs")
    evidence = review["evidence_summary_sha256"]
    if not isinstance(evidence, dict) or set(evidence) != set(required_gates):
        raise CatalogBuildError("review evidence set differs from required gates")
    for kind in required_gates:
        require_string(evidence[kind], f"review evidence {kind}", digest=True)
        if evidence[kind] != evidence_digests[kind]:
            raise CatalogBuildError(f"review evidence digest differs for {kind}")
    reviewers = review["reviewers"]
    if not isinstance(reviewers, dict) or set(reviewers) != set(REVIEW_ROLES):
        raise CatalogBuildError("review must identify all required reviewer roles")
    for role in REVIEW_ROLES:
        require_string(reviewers[role], f"review.reviewers.{role}")
    require_string(review["review_digest"], "review.review_digest", digest=True)
    calculated = _review_digest(review)
    if review["review_digest"] != calculated:
        raise CatalogBuildError("review digest does not match review contents")
    if record["release"]["reviewed_commit"] != reviewed_commit:
        raise CatalogBuildError("record reviewed commit differs")
    if record["release"]["review_digest"] != calculated:
        raise CatalogBuildError("record review digest differs")
    return reviewed_commit


def _existing_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    value = read_object(path, "existing catalog")
    if isinstance(value.get("records"), list):
        records = value["records"]
    elif isinstance(value.get("record"), dict):
        records = [value["record"]]
    else:
        raise CatalogBuildError("existing catalog must contain records or record")
    if any(not isinstance(item, dict) for item in records):
        raise CatalogBuildError("existing catalog records must be objects")
    return records


def _check_existing_records(
    record: dict[str, Any], existing_records: list[dict[str, Any]]
) -> None:
    key = canonical_json({
        name: record[name]
        for name in ("source", "workload", "runtime", "device", "plan")
    })
    for existing in existing_records:
        validate_record_shape(existing)
        existing_key = canonical_json({
            name: existing[name]
            for name in ("source", "workload", "runtime", "device", "plan")
        })
        if existing.get("id") == record["id"]:
            raise CatalogBuildError("existing catalog already contains this record id")
        if existing_key == key:
            raise CatalogBuildError(
                "existing catalog already contains the same source/workload/runtime/device/plan"
            )


def build_record(
    bundle: Path,
    input_path: Path,
    review_path: Path | None,
    performance_bundle: Path | None = None,
    default_bundle: Path | None = None,
    swap_bundle: Path | None = None,
    existing_catalog_path: Path | None = None,
) -> dict[str, Any]:
    record_input = read_object(input_path, "record input")
    record = copy.deepcopy(record_input.get("record", record_input))
    if not isinstance(record, dict):
        raise CatalogBuildError("record input must contain an object")
    provided_digest = record.pop("canonical_record_digest", None)
    validate_record_shape(record)
    if record["release"]["channel"] not in ("staging", "public-experimental", "public-stable"):
        raise CatalogBuildError("record channel is not releasable")
    is_public = record["release"]["channel"].startswith("public-")
    required_gates = PUBLIC_GATES if is_public else STAGING_GATES
    if default_bundle is None:
        raise CatalogBuildError("P0 default-path evidence bundle is required")
    if performance_bundle is None:
        raise CatalogBuildError("P1 same-layout evidence bundle is required")
    if is_public and swap_bundle is None:
        raise CatalogBuildError("P3 swap evidence bundle is required for public release")

    bundle_paths = {
        "P0": default_bundle.resolve(),
        "P1": performance_bundle.resolve(),
        "P2": bundle.resolve(),
    }
    if swap_bundle is not None:
        bundle_paths["P3"] = swap_bundle.resolve()
    summaries: dict[str, dict[str, Any]] = {}
    policies: dict[str, dict[str, Any]] = {}
    evidence_digests: dict[str, str] = {}
    for kind in required_gates:
        summary, policy, digest = _verified_summary(bundle_paths[kind], kind)
        _validate_binding(policy, record, kind)
        if kind in ("P1", "P2", "P3") and (
            record["plan"]["layout_digest"] !=
            _candidate_layout_digest(bundle_paths[kind])
        ):
            raise CatalogBuildError(
                f"record layout digest differs from {kind} candidate receipt"
            )
        summaries[kind] = summary
        policies[kind] = policy
        evidence_digests[kind] = digest

    memory, target, summary_digest = _validate_p2_summary(
        bundle_paths["P2"], summaries["P2"], policies["P2"]
    )
    p3_result = (
        _validate_p3_summary(summaries["P3"], policies["P3"])
        if is_public else None
    )
    if record["calibration"]["calibrated_request_bytes"] > memory["allowed_peak_bytes"]:
        raise CatalogBuildError("record calibrated bytes exceed allowed peak")
    expected_peak = math.ceil(float(memory["peak_p95_bytes"]["candidate"]))
    if record["calibration"]["calibrated_request_bytes"] != expected_peak:
        raise CatalogBuildError(
            "record calibrated bytes do not equal candidate process-tree P95"
        )
    if record["calibration"]["confirmation_sample_count"] != memory["required_count"]:
        raise CatalogBuildError("record confirmation count differs from evidence")
    if record["calibration"]["maximum_sample_gap_ns"] != memory["maximum_sample_gap_ns"]:
        raise CatalogBuildError("record maximum sample gap differs from evidence")
    if record["calibration"]["maximum_sample_gap_ns"] > memory["allowed_max_gap_ns"]:
        raise CatalogBuildError("record maximum sample gap exceeds policy")
    if record["calibration"]["calibrated_request_bytes"] > memory["allowed_peak_bytes"]:
        raise CatalogBuildError("record calibrated bytes exceed public headroom")
    if record["calibration"]["evidence_digest"] != summary_digest:
        raise CatalogBuildError("calibration evidence digest is not summary.json SHA-256")
    performance_digest = evidence_digests["P1"]
    if record["performance"]["evidence_digest"] != performance_digest:
        raise CatalogBuildError(
            "performance evidence digest is not P1 summary.json SHA-256"
        )
    if not review_path:
        raise CatalogBuildError("review is required for a catalog record")
    record_id_digest = record_identity_digest(record)
    review = read_object(review_path, "review")
    reviewed_commit = _validate_review(
        review, record, evidence_digests, record_id_digest, required_gates
    )
    for kind in ("P0", "P2"):
        _validate_commit_identity(
            summaries[kind], reviewed_commit, kind, is_public
        )
    record["id"] = "tc-streaming-" + record_id_digest[:32]
    record["canonical_record_digest"] = canonical_record_digest(record)
    if provided_digest is not None and provided_digest != record["canonical_record_digest"]:
        raise CatalogBuildError("record input canonical digest differs")
    _check_existing_records(record, _existing_records(existing_catalog_path))
    return {
        "schema": SCHEMA,
        "builder_revision": BUILDER_REVISION,
        "status": "verified",
        "record": record,
        "evidence": {
            "summary_sha256": evidence_digests,
            "policy_sha256": {
                kind: summaries[kind].get("policy_sha256")
                for kind in required_gates
            },
            "target_bytes": target,
            "allowed_peak_bytes": memory["allowed_peak_bytes"],
            "required_count": memory["required_count"],
            "maximum_sample_gap_ns": memory["maximum_sample_gap_ns"],
            "allowed_max_gap_ns": memory["allowed_max_gap_ns"],
            "candidate_peak_p95_bytes": expected_peak,
            "fresh_process_generations": memory.get("fresh_process_generations", {}),
            "p3": p3_result,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--default-bundle",
        type=Path,
        help="verified P0 default/resident campaign",
    )
    parser.add_argument(
        "--performance-bundle",
        type=Path,
        help="verified P1 same-layout campaign used for performance evidence",
    )
    parser.add_argument(
        "--swap-bundle",
        type=Path,
        help="verified P3 natural-swap comparison (required for public channels)",
    )
    parser.add_argument(
        "--existing-catalog",
        type=Path,
        help="optional existing catalog snapshot used for duplicate detection",
    )
    parser.add_argument("--record-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_record(
            args.bundle.resolve(), args.record_input.resolve(),
            args.review.resolve() if args.review else None,
            args.performance_bundle.resolve() if args.performance_bundle else None,
            args.default_bundle.resolve() if args.default_bundle else None,
            args.swap_bundle.resolve() if args.swap_bundle else None,
            args.existing_catalog.resolve() if args.existing_catalog else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (CatalogBuildError, OSError, EvidenceError) as exc:
        print(f"catalog builder rejected input: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
