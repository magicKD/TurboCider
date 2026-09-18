#!/usr/bin/env python3
"""Independently verify a TurboCider streaming performance evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from verify_process_tree_samples import (
    EvidenceError as MemoryEvidenceError,
    verify as verify_memory_evidence,
)


SCHEMA_VERSION = 1
VARIANTS = ("baseline", "candidate")
VALID_BLOCK_SEQUENCES = {
    ("baseline", "candidate", "candidate", "baseline"),
    ("candidate", "baseline", "baseline", "candidate"),
}
REQUIRED_P1_SEMANTIC_FIELDS = (
    "stage",
    "resident_prefix_blocks",
    "block_group_size",
    "slot_count",
    "prefetch_distance",
    "io_workers",
    "group_count",
    "pass_count",
    "startup_policy",
    "pass_transition",
    "retention",
    "reader_revision",
    "weight_format",
    "kernel_revision",
    "conditioning_recipe",
    "upsample_boundary",
    "engine_lifecycle",
    "total_fills",
    "request",
)
EXPECTED_P1_DECLARATION_FIELDS = REQUIRED_P1_SEMANTIC_FIELDS[:-1]
KIND_TO_THRESHOLD = {"P0": "P0_legacy", "P1": "P1_same_layout"}
P3_PROTOCOL_REVISION = "tc-p3-natural-swap-v1"
P3_PRESSURE_SOURCES = {
    "natural_low_memory_device",
    "externally_managed_fixed_pressure",
}
SPEC_MAXIMUMS = {
    "P0": {
        "wall_median_ratio_max": 1.02,
        "wall_p95_ratio_max": 1.05,
        "denoise_median_ratio_max": 1.02,
    },
    "P1": {
        "wall_median_ratio_max": 1.02,
        "wall_p95_ratio_max": 1.05,
        "denoise_median_ratio_max": 1.02,
    },
}


class EvidenceError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise EvidenceError(f"cannot read raw samples {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceError(f"raw sample {line_number} is not an object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise EvidenceError(f"{label} must be finite and {qualifier}")
    return value


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise EvidenceError("cannot calculate a percentile over no samples")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def metric(values: list[float], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "count": len(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
    }


def bootstrap_interval(
    blocks: list[list[tuple[float, float]]],
    statistic: str,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    if not blocks or any(not block for block in blocks):
        raise EvidenceError("no complete ABBA blocks for bootstrap")
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        pairs = [pair for block in sampled for pair in block]
        baseline = [item[0] for item in pairs]
        candidate = [item[1] for item in pairs]
        if statistic == "median_ratio":
            values.append(
                statistics.median(candidate) / statistics.median(baseline)
            )
        elif statistic == "p95_ratio":
            values.append(
                percentile(candidate, 0.95) / percentile(baseline, 0.95)
            )
        else:
            raise EvidenceError(f"unknown bootstrap statistic {statistic}")
    return percentile(values, 0.025), percentile(values, 0.975)


def threshold_for(policy: dict[str, Any], metric_name: str) -> float | None:
    kind = policy.get("comparison_kind")
    threshold_name = KIND_TO_THRESHOLD.get(kind)
    if not threshold_name:
        return None
    thresholds = policy.get("thresholds")
    if (
        not isinstance(thresholds, dict) or
        not isinstance(thresholds.get(threshold_name), dict)
    ):
        raise EvidenceError(f"missing thresholds.{threshold_name}")
    value = thresholds[threshold_name].get(metric_name)
    if value is None:
        return None
    threshold = finite_number(value, f"threshold {metric_name}")
    maximum = SPEC_MAXIMUMS[kind][metric_name]
    if threshold > maximum:
        raise EvidenceError(
            f"threshold {metric_name}={threshold} is looser than "
            f"specification {maximum}"
        )
    return threshold


def source_provenance_complete(identity: Any) -> bool:
    def hexadecimal(value: Any, length: int) -> bool:
        return (
            isinstance(value, str) and len(value) == length and
            all(character in "0123456789abcdef" for character in value.lower())
        )

    if not isinstance(identity, dict):
        return False
    source = identity.get("source_identity")
    if (
        not isinstance(source, dict) or
        not hexadecimal(source.get("commit"), 40) or
        not hexadecimal(source.get("source_manifest_sha256"), 64)
    ):
        return False
    if source.get("clean") is True:
        return True
    return bool(
        hexadecimal(source.get("dirty_diff_sha256"), 64) and
        hexadecimal(source.get("untracked_sources_sha256"), 64)
    )


def protocol_complete(policy: dict[str, Any]) -> bool:
    protocol = policy.get("protocol")
    if not isinstance(protocol, dict):
        return False
    required = (
        "measured_blocks", "warmup_requests_per_variant",
        "request_timeout_seconds", "cache_condition", "stopping_rule",
        "quantile_estimator", "bootstrap_unit", "multiplicity_policy",
        "exclusion_policy", "timing_scope",
    )
    if any(key not in protocol for key in required):
        return False
    if protocol.get("bootstrap_unit") != "abba_block":
        return False
    return bool(
        policy.get("bootstrap_iterations") and
        policy.get("bootstrap_seed") is not None
    )


def p3_contract(policy: dict[str, Any]) -> dict[str, Any]:
    value = policy.get("swap_comparison")
    required = {
        "revision", "pressure_source", "baseline_role", "candidate_role",
        "minimum_baseline_swap_runs", "candidate_swap_out_total_ratio_max",
        "reporting_mode",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise EvidenceError(
            "P3 swap_comparison must contain the exact natural-swap contract"
        )
    if value.get("revision") != P3_PROTOCOL_REVISION:
        raise EvidenceError("P3 swap comparison revision is unsupported")
    if value.get("pressure_source") not in P3_PRESSURE_SOURCES:
        raise EvidenceError("P3 pressure source is unsupported")
    if value.get("baseline_role") != "resident_or_default":
        raise EvidenceError("P3 baseline role must be resident_or_default")
    if value.get("candidate_role") != "public_streaming_exact":
        raise EvidenceError("P3 candidate role must be public_streaming_exact")
    minimum_runs = value.get("minimum_baseline_swap_runs")
    if (
        isinstance(minimum_runs, bool) or not isinstance(minimum_runs, int) or
        not 1 <= minimum_runs <= 20
    ):
        raise EvidenceError(
            "P3 minimum_baseline_swap_runs must be an integer in 1...20"
        )
    maximum_ratio = finite_number(
        value.get("candidate_swap_out_total_ratio_max"),
        "P3 candidate swap-out ratio maximum",
    )
    if maximum_ratio > 1:
        raise EvidenceError(
            "P3 candidate swap-out ratio maximum cannot exceed 1"
        )
    if value.get("reporting_mode") != "tradeoff_or_speedup":
        raise EvidenceError("P3 reporting_mode must be tradeoff_or_speedup")
    return value


def validate_p3_environment(
    policy: dict[str, Any], environment: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    contract = p3_contract(policy)
    pressure = environment.get("pressure")
    required = {
        "protocol_revision", "source", "state", "launched_by_runner",
        "cleanup_verified", "counter_scope",
    }
    if not isinstance(pressure, dict) or set(pressure) != required:
        raise EvidenceError(
            "P3 environment.pressure must contain the exact pressure record"
        )
    if pressure.get("protocol_revision") != P3_PROTOCOL_REVISION:
        raise EvidenceError("P3 environment pressure revision differs")
    if pressure.get("source") != contract["pressure_source"]:
        raise EvidenceError("P3 environment pressure source differs from policy")
    if pressure.get("state") != "stable":
        raise EvidenceError("P3 pressure state was not stable")
    if pressure.get("launched_by_runner") is not False:
        raise EvidenceError("P3 runner must not launch memory pressure")
    if manifest.get("pressure_launched_by_runner") is not False:
        raise EvidenceError("P3 manifest reports runner-launched pressure")
    if pressure.get("cleanup_verified") is not True:
        raise EvidenceError("P3 pressure cleanup was not verified")
    if pressure.get("counter_scope") != "host_global":
        raise EvidenceError("P3 pressure counters must declare host_global scope")
    return contract


def evaluate_p3_swap(
    contract: dict[str, Any],
    swap_out_total_bytes: dict[str, int],
    swap_out_run_count: dict[str, int],
) -> tuple[str, float | None]:
    baseline_total = swap_out_total_bytes.get("baseline", 0)
    candidate_total = swap_out_total_bytes.get("candidate", 0)
    baseline_runs = swap_out_run_count.get("baseline", 0)
    if baseline_total <= 0 or baseline_runs < contract[
        "minimum_baseline_swap_runs"
    ]:
        return "INCONCLUSIVE", None
    ratio = candidate_total / baseline_total
    return (
        "PASS" if ratio <= contract[
            "candidate_swap_out_total_ratio_max"
        ] else "FAIL",
        ratio,
    )


def audit_state(audit: dict[str, Any]) -> str:
    status = audit.get("status")
    if audit.get("passed") is True and status in ("passed", "complete"):
        return "passed"
    if audit.get("passed") is False or status == "failed":
        return "failed"
    return "partial"


def audit_contract(
    policy: dict[str, Any], audit: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    state = audit_state(audit)
    kind = policy.get("comparison_kind")
    fields = {
        "P0": (
            "new_framework_hooks", "new_memory_probes",
            "new_worker_threads", "new_pool_allocations",
            "new_cache_clear_or_unload_calls",
        ),
        "P1": (
            "steady_framework_allocations",
            "steady_framework_thread_creates",
        ),
    }.get(kind, ())
    threshold_name = KIND_TO_THRESHOLD.get(kind)
    threshold_group = policy.get("thresholds", {}).get(threshold_name, {})
    observations: dict[str, Any] = {}
    if state != "passed":
        return state, observations
    for field in fields:
        limit = threshold_group.get(field)
        value = audit.get(field)
        if (
            isinstance(limit, bool) or not isinstance(limit, (int, float)) or
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return "partial", observations
        observations[field] = {"value": value, "maximum": limit}
        if value > limit:
            return "failed", observations
    return "passed", observations


def validate_manifest(
    bundle: Path,
    manifest: dict[str, Any],
    policy_sha256: str,
    comparison_kind: str,
) -> None:
    if manifest.get("policy_sha256") != policy_sha256:
        raise EvidenceError("campaign policy hash does not match manifest")
    files = manifest.get("files")
    if files is None:
        return
    if not isinstance(files, dict):
        raise EvidenceError("manifest.files must be an object")
    required = [
        "campaign-policy.json", "build-identity.json", "raw-samples.jsonl",
        "warmups.jsonl", "quality.json", "audit.json", "faults.json",
        "environment.json",
    ]
    if comparison_kind == "P1":
        required.append("semantic-equivalence.json")
    memory_sampling = read_json(bundle / "campaign-policy.json").get(
        "memory_sampling"
    )
    if isinstance(memory_sampling, dict) and memory_sampling.get("enabled") is True:
        required.append("memory-summaries.jsonl")
    for name in required:
        validate_manifest_file(bundle, files, name)


def bundle_file(bundle: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise EvidenceError(f"{label} must be a relative bundle path")
    path = (bundle / value).resolve()
    try:
        relative = str(path.relative_to(bundle))
    except ValueError as exc:
        raise EvidenceError(f"{label} escapes the evidence bundle") from exc
    return path, relative


def validate_manifest_file(
    bundle: Path, files: dict[str, Any], name: str
) -> None:
    record = files.get(name)
    if not isinstance(record, dict) or not record.get("sha256"):
        raise EvidenceError(f"manifest lacks hash for {name}")
    path, relative = bundle_file(bundle, name, f"manifest file {name}")
    if relative != name or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise EvidenceError(f"manifest hash mismatch for {name}")


def parse_samples(
    raw: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    blocks: dict[str, list[dict[str, Any]]] = {}
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    positions: set[tuple[str, int]] = set()
    expected_implementations = policy.get("expected_implementations")
    if expected_implementations is not None and (
        not isinstance(expected_implementations, dict) or
        set(expected_implementations) != set(VARIANTS) or
        any(
            not isinstance(expected_implementations[name], str) or
            not expected_implementations[name]
            for name in VARIANTS
        )
    ):
        raise EvidenceError(
            "expected_implementations must contain non-empty baseline and "
            "candidate strings"
        )
    for index, row in enumerate(raw):
        block_id = row.get("block_id")
        pair_id = row.get("pair_id")
        variant = row.get("variant")
        position = row.get("position")
        if not isinstance(block_id, str) or not block_id:
            raise EvidenceError(f"raw sample {index} has no block_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise EvidenceError(f"raw sample {index} has no pair_id")
        if variant not in VARIANTS:
            raise EvidenceError(f"raw sample {index} has invalid variant")
        if (
            isinstance(position, bool) or not isinstance(position, int) or
            not 0 <= position < 4
        ):
            raise EvidenceError(f"raw sample {index} has invalid block position")
        key = (block_id, position)
        if key in positions:
            raise EvidenceError(f"duplicate position {position} in block {block_id}")
        positions.add(key)
        blocks.setdefault(block_id, []).append(row)
        variants = pairs.setdefault(pair_id, {})
        if variant in variants:
            raise EvidenceError(f"duplicate {variant} sample for pair {pair_id}")
        status = row.get("status", "success")
        parsed = {"row": row, "status": status, "index": index}
        if status == "success":
            expected_lifecycle = policy.get(
                "engine_lifecycle", "persistent"
            )
            observed_lifecycle = row.get(
                "engine_lifecycle", "persistent"
            )
            if observed_lifecycle != expected_lifecycle:
                raise EvidenceError(
                    f"sample {index} engine lifecycle differs from policy"
                )
            parsed.update({
                "wall": finite_number(
                    row.get("request_wall_seconds"),
                    f"sample {index} wall", positive=True,
                ),
                "denoise": finite_number(
                    row.get("denoise_seconds"),
                    f"sample {index} denoise", positive=True,
                ),
                "resolved_layout": row.get("resolved_layout_digest"),
                "actual_layout": row.get("actual_layout_digest"),
                "actual_semantic_layout": row.get(
                    "actual_semantic_layout"
                ),
                "actual_semantic_layout_digest": row.get(
                    "actual_semantic_layout_digest"
                ),
                "actual_semantic_layout_missing_fields": row.get(
                    "actual_semantic_layout_missing_fields", []
                ),
                "streaming_implementation": row.get(
                    "streaming_implementation"
                ),
            })
            if expected_implementations is not None:
                expected_implementation = expected_implementations[variant]
                if (
                    parsed["streaming_implementation"] !=
                    expected_implementation
                ):
                    raise EvidenceError(
                        f"sample {index} {variant} streaming implementation "
                        f"differs: expected {expected_implementation!r}, got "
                        f"{parsed['streaming_implementation']!r}"
                    )
            semantic = parsed["actual_semantic_layout"]
            semantic_sha256 = parsed["actual_semantic_layout_digest"]
            if isinstance(semantic, dict):
                if semantic_digest(semantic) != semantic_sha256:
                    raise EvidenceError(
                        f"sample {index} semantic layout digest mismatch"
                    )
                if policy.get("comparison_kind") == "P1":
                    missing = [
                        field for field in REQUIRED_P1_SEMANTIC_FIELDS
                        if field not in semantic
                    ]
                    if missing:
                        raise EvidenceError(
                            f"sample {index} semantic layout is incomplete: "
                            + ", ".join(missing)
                        )
            elif semantic_sha256 is not None:
                raise EvidenceError(
                    f"sample {index} has a semantic digest without a layout"
                )
            if (
                parsed["resolved_layout"] is not None and
                parsed["actual_layout"] is not None and
                parsed["resolved_layout"] != parsed["actual_layout"]
            ):
                raise EvidenceError(
                    f"sample {index} resolved/actual layout mismatch"
                )
        else:
            failures.append({
                "index": index,
                "block_id": block_id,
                "pair_id": pair_id,
                "variant": variant,
                "status": status,
                "error": row.get("error"),
            })
        variants[variant] = parsed
    expected_blocks = int(policy.get("protocol", {}).get("measured_blocks", 0))
    if expected_blocks and len(blocks) != expected_blocks:
        raise EvidenceError(
            f"raw samples contain {len(blocks)} blocks, expected {expected_blocks}"
        )
    for block_id, rows in blocks.items():
        if len(rows) != 4 or {row["position"] for row in rows} != {0, 1, 2, 3}:
            raise EvidenceError(
                f"block {block_id} is not a complete four-position block"
            )
        ordered = sorted(rows, key=lambda row: row["position"])
        variants = tuple(row["variant"] for row in ordered)
        if variants not in VALID_BLOCK_SEQUENCES:
            raise EvidenceError(f"block {block_id} is not ABBA or BAAB")
        if ordered[0]["pair_id"] != ordered[1]["pair_id"]:
            raise EvidenceError(f"block {block_id} first pair is not adjacent")
        if ordered[2]["pair_id"] != ordered[3]["pair_id"]:
            raise EvidenceError(f"block {block_id} second pair is not adjacent")
        if ordered[0]["pair_id"] == ordered[2]["pair_id"]:
            raise EvidenceError(f"block {block_id} reuses one pair ID")
        for first in (0, 2):
            if {
                ordered[first]["variant"], ordered[first + 1]["variant"]
            } != set(VARIANTS):
                raise EvidenceError(f"block {block_id} pair lacks both variants")
    for pair_id, variants in pairs.items():
        if set(variants) != set(VARIANTS):
            raise EvidenceError(f"pair {pair_id} lacks baseline or candidate")
        block_ids = {item["row"]["block_id"] for item in variants.values()}
        if len(block_ids) != 1:
            raise EvidenceError(f"pair {pair_id} crosses block boundaries")
    return blocks, pairs, failures


def verify_memory_bundle(
    bundle: Path,
    policy: dict[str, Any],
    raw: list[dict[str, Any]],
    warmups: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    config = policy.get("memory_sampling")
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return None
    required_variants = config.get("required_variants", ["candidate"])
    if (
        not isinstance(required_variants, list) or not required_variants or
        len(set(required_variants)) != len(required_variants) or
        any(value not in VARIANTS for value in required_variants)
    ):
        raise EvidenceError("memory_sampling.required_variants is invalid")
    comparison_kind = policy.get("comparison_kind")
    if comparison_kind == "P2":
        public_targets = {value << 30 for value in (8, 10, 12, 16, 20)}
        if config.get("target_bytes") not in public_targets:
            raise EvidenceError("P2 target is not a public memory tier")
        if config.get("headroom_policy_revision") != "tc-public-headroom-v1":
            raise EvidenceError("P2 headroom policy revision is invalid")
        if "candidate" not in required_variants:
            raise EvidenceError("P2 candidate memory evidence is not required")
    if comparison_kind == "P3":
        p3_contract(policy)
        if required_variants != ["baseline", "candidate"]:
            raise EvidenceError(
                "P3 memory evidence must require baseline and candidate in order"
            )
        if config.get("allow_swap_out") is not True:
            raise EvidenceError("P3 memory sampling must allow observed swap-out")
        if config.get("include_warmups") is True:
            raise EvidenceError(
                "P3 memory evidence must exclude warmups from swap comparison"
            )
        if config.get("target_bytes") is not None:
            raise EvidenceError("P3 natural-swap evidence must not apply a P2 target")
    path = bundle / "memory-summaries.jsonl"
    try:
        rows = read_jsonl(path)
    except EvidenceError as exc:
        raise EvidenceError(f"cannot read memory-summaries.jsonl: {exc}") from exc
    if not rows:
        raise EvidenceError("memory-summaries.jsonl is empty")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise EvidenceError("memory evidence requires manifest file hashes")
    by_run: dict[str, dict[str, Any]] = {}

    def root_identity(value: Any, label: str) -> tuple[int, int, int]:
        if not isinstance(value, dict):
            raise EvidenceError(f"{label} root identity is missing")
        result = []
        for field in ("pid", "start_seconds", "start_microseconds"):
            item = value.get(field)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise EvidenceError(f"{label}.{field} is invalid")
            result.append(item)
        return result[0], result[1], result[2]

    for index, row in enumerate(rows):
        if row.get("schema") != "turbocider-streaming-memory-summary-v1":
            raise EvidenceError(f"memory summary {index} has invalid schema")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in by_run:
            raise EvidenceError(
                f"memory summary {index} has a duplicate or invalid run_id"
            )
        phase = row.get("phase")
        if phase not in ("warmup", "measured"):
            raise EvidenceError(f"memory summary {index} has invalid phase")
        variant = row.get("variant")
        if variant not in VARIANTS:
            raise EvidenceError(f"memory summary {index} has invalid variant")
        root_identity(row.get("root_identity"), f"memory summary {index}")
        evidence_path, relative = bundle_file(
            bundle, row.get("evidence_path"),
            f"memory summary {index}.evidence_path",
        )
        if relative != row.get("evidence_path"):
            raise EvidenceError("memory evidence path is not canonical")
        validate_manifest_file(bundle, manifest_files, relative)
        summary_path, summary_relative = bundle_file(
            bundle, row.get("summary_path"),
            f"memory summary {index}.summary_path",
        )
        if summary_relative != row.get("summary_path"):
            raise EvidenceError("memory summary path is not canonical")
        validate_manifest_file(bundle, manifest_files, summary_relative)
        persisted_summary = read_json(summary_path)
        if persisted_summary != row:
            raise EvidenceError(
                f"persisted memory summary differs for {run_id}"
            )
        try:
            evidence = verify_memory_evidence(evidence_path)
        except MemoryEvidenceError as exc:
            raise EvidenceError(
                f"memory evidence for {run_id} is invalid: {exc}"
            ) from exc
        if row.get("evidence_digest") != evidence["final_evidence_digest"]:
            raise EvidenceError(f"memory evidence digest differs for {run_id}")
        for field in (
            "sampler_revision", "correlation_id", "status", "complete",
            "root_identity",
            "sample_count", "max_gap_ns",
            "allowed_max_gap_ns", "tree_peak_rss_bytes",
            "tree_peak_phys_footprint_bytes", "swap_in_bytes", "swap_out_bytes",
            "compression_bytes", "decompression_bytes", "reasons",
            "command_exit_code",
        ):
            if row.get(field) != evidence.get(field):
                raise EvidenceError(
                    f"memory summary {run_id} differs from independent verifier"
                )
        by_run[run_id] = row

    expected_rows = [
        *(
            row for row in warmups
            if config.get("include_warmups") is True
        ),
        *raw,
    ]
    required_run_ids = {
        row["run_id"] for row in expected_rows
        if row.get("variant") in required_variants
    }
    missing = sorted(required_run_ids - set(by_run))
    if missing:
        raise EvidenceError(
            "missing process-tree memory summaries: " + ", ".join(missing)
        )
    unexpected = sorted(set(by_run) - {row["run_id"] for row in expected_rows})
    if unexpected:
        raise EvidenceError(
            "unexpected process-tree memory summaries: " + ", ".join(unexpected)
        )
    expected_by_run = {
        row["run_id"]: (phase, row)
        for phase, phase_rows in (
            ("warmup", warmups if config.get("include_warmups") is True else []),
            ("measured", raw),
        )
        for row in phase_rows
    }
    for run_id in sorted(by_run):
        expected_phase, expected_row = expected_by_run[run_id]
        summary = by_run[run_id]
        if (
            summary.get("phase") != expected_phase or
            summary.get("variant") != expected_row.get("variant") or
            summary.get("correlation_id") != run_id or
            expected_row.get("memory_summary") != summary
        ):
            raise EvidenceError(
                f"memory summary identity differs for {run_id}"
            )
        if expected_phase == "measured" and any(
            summary.get(field) != expected_row.get(field)
            for field in (
                "block_id", "block_index", "pair_id", "pair_index", "position"
            )
        ):
            raise EvidenceError(
                f"memory summary block identity differs for {run_id}"
            )
    incomplete = sorted(
        run_id for run_id in required_run_ids
        if by_run[run_id].get("complete") is not True or
        by_run[run_id].get("status") != "complete"
    )
    target = config.get("target_bytes")
    allowed_peak = None
    over_target: list[str] = []
    required_peaks = {
        variant: [
            by_run[row["run_id"]]["tree_peak_phys_footprint_bytes"]
            for row in expected_rows
            if row.get("variant") == variant and row["run_id"] in required_run_ids
        ]
        for variant in required_variants
    }
    peak_p95 = {
        variant: percentile([float(value) for value in values], 0.95)
        for variant, values in required_peaks.items() if values
    }
    insufficient_variants = sorted(
        variant for variant, values in required_peaks.items()
        if comparison_kind in ("P2", "P3") and len(values) < 20
    )
    fresh_process_failures: list[str] = []
    fresh_process_generations: dict[str, int] = {}
    if comparison_kind in ("P2", "P3"):
        expected_blocks = int(policy.get("protocol", {}).get("measured_blocks", 0))
        for variant in required_variants:
            identities_by_block: dict[int, set[tuple[int, int, int]]] = {}
            for row in raw:
                if row.get("variant") != variant:
                    continue
                summary = by_run[row["run_id"]]
                identities_by_block.setdefault(row["block_index"], set()).add(
                    root_identity(summary.get("root_identity"), row["run_id"])
                )
            identities = {
                identity for values in identities_by_block.values()
                for identity in values
            }
            fresh_process_generations[variant] = len(identities)
            if (
                len(identities_by_block) != expected_blocks or
                any(len(values) != 1 for values in identities_by_block.values()) or
                len(identities) != expected_blocks or len(identities) < 2
            ):
                fresh_process_failures.append(variant)
    if target is not None:
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise EvidenceError("memory_sampling.target_bytes is invalid")
        margin = max(512 << 20, (target * 10 + 99) // 100)
        allowed_peak = target - margin
        if allowed_peak <= 0:
            raise EvidenceError("memory_sampling target leaves no allowed peak")
        over_target = sorted(
            run_id for run_id in required_run_ids
            if by_run[run_id]["tree_peak_phys_footprint_bytes"] > allowed_peak
        )
    unexpected_swap = sorted(
        run_id for run_id in required_run_ids
        if by_run[run_id]["swap_out_bytes"] != 0
    )
    disallowed_swap = (
        unexpected_swap if config.get("allow_swap_out") is not True else []
    )
    required_count_by_variant = {
        variant: len(values) for variant, values in required_peaks.items()
    }
    swap_out_total_bytes = {
        variant: sum(
            by_run[row["run_id"]]["swap_out_bytes"]
            for row in expected_rows
            if row.get("variant") == variant and row["run_id"] in required_run_ids
        )
        for variant in required_variants
    }
    swap_in_total_bytes = {
        variant: sum(
            by_run[row["run_id"]]["swap_in_bytes"]
            for row in expected_rows
            if row.get("variant") == variant and row["run_id"] in required_run_ids
        )
        for variant in required_variants
    }
    compression_total_bytes = {
        variant: sum(
            by_run[row["run_id"]]["compression_bytes"]
            for row in expected_rows
            if row.get("variant") == variant and row["run_id"] in required_run_ids
        )
        for variant in required_variants
    }
    decompression_total_bytes = {
        variant: sum(
            by_run[row["run_id"]]["decompression_bytes"]
            for row in expected_rows
            if row.get("variant") == variant and row["run_id"] in required_run_ids
        )
        for variant in required_variants
    }
    swap_out_run_count = {
        variant: sum(
            by_run[row["run_id"]]["swap_out_bytes"] > 0
            for row in expected_rows
            if row.get("variant") == variant and row["run_id"] in required_run_ids
        )
        for variant in required_variants
    }
    p3_swap_status = "NOT_APPLICABLE"
    candidate_swap_out_total_ratio = None
    if comparison_kind == "P3":
        contract = p3_contract(policy)
        p3_swap_status, candidate_swap_out_total_ratio = evaluate_p3_swap(
            contract, swap_out_total_bytes, swap_out_run_count
        )
    maximum_sample_gap_ns = max(
        (by_run[run_id]["max_gap_ns"] for run_id in required_run_ids),
        default=0,
    )
    allowed_max_gap_ns = min(
        (by_run[run_id]["allowed_max_gap_ns"] for run_id in required_run_ids),
        default=0,
    )
    if over_target or disallowed_swap or p3_swap_status == "FAIL":
        qualification = "FAIL"
    elif (
        incomplete or insufficient_variants or fresh_process_failures or
        p3_swap_status == "INCONCLUSIVE"
    ):
        qualification = "INCONCLUSIVE"
    else:
        qualification = "PASS"
    return {
        "enabled": True,
        "qualification": qualification,
        "summary_count": len(by_run),
        "required_count": len(required_run_ids),
        "required_count_by_variant": required_count_by_variant,
        "required_variants": required_variants,
        "target_bytes": target,
        "allowed_peak_bytes": allowed_peak,
        "peak_p95_bytes": peak_p95,
        "maximum_sample_gap_ns": maximum_sample_gap_ns,
        "allowed_max_gap_ns": allowed_max_gap_ns,
        "over_target": over_target,
        "unexpected_swap": unexpected_swap,
        "swap_out_total_bytes": swap_out_total_bytes,
        "swap_in_total_bytes": swap_in_total_bytes,
        "compression_total_bytes": compression_total_bytes,
        "decompression_total_bytes": decompression_total_bytes,
        "swap_out_run_count": swap_out_run_count,
        "p3_swap_status": p3_swap_status,
        "candidate_swap_out_total_ratio": candidate_swap_out_total_ratio,
        "incomplete": incomplete,
        "insufficient_variants": insufficient_variants,
        "fresh_process_generations": fresh_process_generations,
        "fresh_process_failures": fresh_process_failures,
    }


def verify(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    policy_path = bundle / "campaign-policy.json"
    policy = read_json(policy_path)
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("unsupported or missing campaign policy schema_version")
    if policy.get("status") not in ("frozen", "executed"):
        raise EvidenceError("campaign policy must be frozen or executed")
    comparison_kind = policy.get("comparison_kind")
    if comparison_kind not in ("P0", "P1", "P2", "P3", "P4"):
        raise EvidenceError("comparison_kind must be P0, P1, P2, P3 or P4")
    if comparison_kind == "P2" and (
        policy.get("engine_lifecycle", "persistent") != "per_request" or
        policy.get("protocol", {}).get("restart_workers_between_blocks") is not True
    ):
        raise EvidenceError(
            "P2 requires per_request engines and worker restart between blocks"
        )
    raw = read_jsonl(bundle / "raw-samples.jsonl")
    warmups = read_jsonl(bundle / "warmups.jsonl")
    quality = read_json(bundle / "quality.json")
    build = read_json(bundle / "build-identity.json")
    audit = read_json(bundle / "audit.json")
    environment = read_json(bundle / "environment.json")
    faults = read_json(bundle / "faults.json")
    manifest = read_json(bundle / "manifest.json")
    if not raw:
        raise EvidenceError("raw-samples.jsonl is empty")
    if not isinstance(quality.get("samples"), list):
        raise EvidenceError("quality.json must contain samples")
    if (
        not isinstance(build.get("baseline"), dict) or
        not isinstance(build.get("candidate"), dict)
    ):
        raise EvidenceError(
            "build-identity.json must identify baseline and candidate"
        )
    policy_sha256 = sha256_file(policy_path)
    validate_manifest(bundle, manifest, policy_sha256, comparison_kind)
    p3_environment = None
    if comparison_kind == "P3":
        p3_environment = validate_p3_environment(policy, environment, manifest)
    memory_evidence = verify_memory_bundle(
        bundle, policy, raw, warmups, manifest
    )
    baseline_identity = build["baseline"]
    candidate_identity = build["candidate"]
    if comparison_kind == "P0" and (
        not baseline_identity.get("binary_sha256") or
        not candidate_identity.get("binary_sha256") or
        baseline_identity.get("binary_sha256") ==
        candidate_identity.get("binary_sha256")
    ):
        raise EvidenceError(
            "P0 requires distinct baseline/candidate binary identities"
        )
    provenance_ok = (
        source_provenance_complete(baseline_identity) and
        source_provenance_complete(candidate_identity)
    ) if comparison_kind in ("P0", "P2", "P3") else True
    if comparison_kind == "P1":
        semantic = read_json(bundle / "semantic-equivalence.json")
        if semantic.get("format") != (
            "turbocider-streaming-semantic-equivalence-v2"
        ):
            raise EvidenceError(
                "P1 requires normalized semantic-equivalence v2 evidence"
            )
        declaration = policy.get("semantic_equivalence")
        expected = (
            declaration.get("expected_actual")
            if isinstance(declaration, dict) else None
        )
        if not isinstance(expected, dict) or any(
            field not in expected
            for field in EXPECTED_P1_DECLARATION_FIELDS
        ):
            raise EvidenceError(
                "P1 policy lacks a complete expected actual layout"
            )
        if (
            semantic.get("expected_actual") != expected or
            semantic.get("observed_matches_expected") is not True
        ):
            raise EvidenceError(
                "P1 observed semantics differ from the frozen expected layout"
            )
        if semantic.get("equivalent") is not True:
            raise EvidenceError(
                "P1 requires observed and declared semantic-equivalence evidence"
            )

    blocks, pairs, failures = parse_samples(raw, policy)
    if faults.get("status") != "complete" or not isinstance(
        faults.get("samples"), list
    ):
        raise EvidenceError("faults.json must be a complete sample list")
    expected_faults = sorted(
        ("measured", item.get("run_id"), item.get("status"))
        for item in raw if item.get("status") != "success"
    ) + sorted(
        ("warmup", item.get("run_id"), item.get("status"))
        for item in warmups if item.get("status") != "success"
    )
    recorded_faults = sorted(
        (item.get("phase"), item.get("run_id"), item.get("status"))
        for item in faults["samples"] if isinstance(item, dict)
    )
    if (
        sorted(expected_faults) != recorded_faults or
        faults.get("failure_count") != len(recorded_faults)
    ):
        raise EvidenceError("faults.json does not match failure samples")
    quality_by_pair: dict[str, dict[str, Any]] = {}
    for item in quality["samples"]:
        if not isinstance(item, dict) or not isinstance(item.get("pair_id"), str):
            raise EvidenceError("quality sample lacks pair_id")
        if item["pair_id"] in quality_by_pair:
            raise EvidenceError(
                f"duplicate quality sample for {item['pair_id']}"
            )
        quality_by_pair[item["pair_id"]] = item
    missing_quality = sorted(set(pairs) - set(quality_by_pair))
    if missing_quality:
        raise EvidenceError(
            f"missing quality records for pairs: {', '.join(missing_quality)}"
        )
    measured_pair_ids = {
        pair_id for pair_id, variants in pairs.items()
        if all(variants[name]["status"] == "success" for name in VARIANTS)
    }
    quality_failures = sorted(
        pair_id for pair_id in measured_pair_ids
        if quality_by_pair[pair_id].get("passed") is not True
    )
    quality_not_measured = sorted(set(pairs) - measured_pair_ids)

    successful_blocks: dict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = {}
    for block_id, rows in blocks.items():
        pair_ids: list[str] = []
        for row in sorted(rows, key=lambda item: item["position"]):
            if row["pair_id"] not in pair_ids:
                pair_ids.append(row["pair_id"])
        block_pairs = []
        for pair_id in pair_ids:
            variants = pairs[pair_id]
            if all(variants[name]["status"] == "success" for name in VARIANTS):
                baseline = variants["baseline"]
                candidate = variants["candidate"]
                if comparison_kind == "P1":
                    left = baseline.get("actual_semantic_layout")
                    right = candidate.get("actual_semantic_layout")
                    left_digest = baseline.get(
                        "actual_semantic_layout_digest"
                    )
                    right_digest = candidate.get(
                        "actual_semantic_layout_digest"
                    )
                    left_missing = baseline.get(
                        "actual_semantic_layout_missing_fields"
                    )
                    right_missing = candidate.get(
                        "actual_semantic_layout_missing_fields"
                    )
                    if (
                        not isinstance(left, dict) or
                        not isinstance(right, dict) or
                        left != right or
                        not left_digest or
                        left_digest != right_digest or
                        left_missing or right_missing
                    ):
                        raise EvidenceError(
                            f"P1 pair {pair_id} lacks an identical complete "
                            "actual semantic layout"
                        )
                block_pairs.append((baseline, candidate))
        if len(block_pairs) == 2:
            successful_blocks[block_id] = block_pairs

    wall_blocks = [
        [(left["wall"], right["wall"]) for left, right in block]
        for block in successful_blocks.values()
    ]
    denoise_blocks = [
        [(left["denoise"], right["denoise"]) for left, right in block]
        for block in successful_blocks.values()
    ]
    wall_pairs = [pair for block in wall_blocks for pair in block]

    minimum_pairs = int(policy.get("initial_matched_pairs", 0))
    if minimum_pairs <= 0:
        raise EvidenceError("initial_matched_pairs must be positive")
    tail_minimum = int(policy.get("minimum_tail_requests_per_variant", 0))
    required_pairs = max(minimum_pairs, tail_minimum)
    sample_sufficient = len(wall_pairs) >= required_pairs
    iterations = int(policy.get("bootstrap_iterations", 4000))
    if iterations < 1000:
        raise EvidenceError("bootstrap_iterations must be at least 1000")
    seed = int(policy.get("bootstrap_seed", 17))
    metrics: dict[str, Any] = {}
    if wall_pairs:
        for name, block_values in (
            ("wall", wall_blocks), ("denoise", denoise_blocks)
        ):
            values = [pair for block in block_values for pair in block]
            baseline = [left for left, _ in values]
            candidate = [right for _, right in values]
            median_ratio = (
                statistics.median(candidate) / statistics.median(baseline)
            )
            p95_ratio = percentile(candidate, 0.95) / percentile(baseline, 0.95)
            median_ci = bootstrap_interval(
                block_values, "median_ratio", seed, iterations
            )
            p95_ci = bootstrap_interval(
                block_values, "p95_ratio", seed + 1, iterations
            )
            metrics[name] = {
                "baseline": metric(baseline, f"baseline_{name}"),
                "candidate": metric(candidate, f"candidate_{name}"),
                "ratio": {"median": median_ratio, "p95": p95_ratio},
                "bootstrap_95": {
                    "median_ratio": {
                        "lower": median_ci[0], "upper": median_ci[1]
                    },
                    "p95_ratio": {
                        "lower": p95_ci[0], "upper": p95_ci[1]
                    },
                },
            }

    audit_status, audit_observations = audit_contract(policy, audit)
    quality_complete = quality.get("status") == "complete"
    environment_complete = environment.get("status") == "complete"
    campaign_complete = manifest.get("status") == "complete"
    protocol_ok = protocol_complete(policy)
    memory_complete = (
        memory_evidence is None or
        memory_evidence.get("qualification") == "PASS"
    )
    memory_failed = (
        memory_evidence is not None and
        memory_evidence.get("qualification") == "FAIL"
    )
    p3_environment_complete = (
        comparison_kind != "P3" or p3_environment is not None
    )
    prerequisites_complete = (
        audit_status == "passed" and quality_complete and provenance_ok and
        environment_complete and campaign_complete and protocol_ok and
        p3_environment_complete and sample_sufficient and memory_complete and not failures and
        not quality_failures
    )
    wall_threshold = threshold_for(policy, "wall_median_ratio_max")
    p95_threshold = threshold_for(policy, "wall_p95_ratio_max")
    denoise_threshold = threshold_for(policy, "denoise_median_ratio_max")
    decisions: dict[str, str] = {}
    decision_specs = (
        ("wall_median", "wall", wall_threshold, "median_ratio"),
        ("wall_p95", "wall", p95_threshold, "p95_ratio"),
        ("denoise_median", "denoise", denoise_threshold, "median_ratio"),
    )
    for label, section_name, threshold, ci_key in decision_specs:
        if threshold is None:
            decisions[label] = "NOT_APPLICABLE"
            continue
        if section_name not in metrics:
            decisions[label] = "NOT_EVALUATED"
            continue
        section = metrics[section_name]
        upper = section["bootstrap_95"][ci_key]["upper"]
        lower = section["bootstrap_95"][ci_key]["lower"]
        if lower > threshold:
            decisions[label] = "FAIL"
        elif prerequisites_complete and upper <= threshold:
            decisions[label] = "PASS"
        else:
            decisions[label] = "INCONCLUSIVE"

    total_requests = len(raw)
    successful_requests = sum(row.get("status") == "success" for row in raw)
    hard_failure_statuses = {"failure", "timeout", "cancelled", "worker_error"}
    hard_failure_samples = [
        item for item in failures if item["status"] in hard_failure_statuses
    ]
    hard_failure = bool(
        hard_failure_samples or quality_failures or audit_status == "failed" or
        memory_failed
    )
    unsupported_kind = comparison_kind not in (*KIND_TO_THRESHOLD, "P2", "P3")
    if hard_failure or any(value == "FAIL" for value in decisions.values()):
        overall = "FAIL"
    elif unsupported_kind or not prerequisites_complete:
        overall = "INCONCLUSIVE"
    else:
        overall = (
            "PASS" if all(
                value in ("PASS", "NOT_APPLICABLE")
                for value in decisions.values()
            ) else "INCONCLUSIVE"
        )
    p3_result = None
    if comparison_kind == "P3" and memory_evidence is not None:
        wall = metrics.get("wall", {})
        denoise = metrics.get("denoise", {})
        wall_ci = wall.get("bootstrap_95", {})
        denoise_ci = denoise.get("bootstrap_95", {})
        faster = bool(
            prerequisites_complete and
            wall_ci.get("median_ratio", {}).get("upper", math.inf) <= 1 and
            wall_ci.get("p95_ratio", {}).get("upper", math.inf) <= 1 and
            denoise_ci.get("median_ratio", {}).get("upper", math.inf) <= 1
        )
        p3_result = {
            "qualification": memory_evidence.get("qualification"),
            "classification": (
                "faster_and_lower_swap" if faster else
                "lower_swap_tradeoff" if memory_evidence.get(
                    "qualification"
                ) == "PASS" else "not_qualified"
            ),
            "speedup_claim_qualified": faster,
            "swap_status": memory_evidence.get("p3_swap_status"),
            "candidate_swap_out_total_ratio": memory_evidence.get(
                "candidate_swap_out_total_ratio"
            ),
            "swap_out_total_bytes": memory_evidence.get(
                "swap_out_total_bytes"
            ),
            "swap_in_total_bytes": memory_evidence.get(
                "swap_in_total_bytes"
            ),
            "compression_total_bytes": memory_evidence.get(
                "compression_total_bytes"
            ),
            "decompression_total_bytes": memory_evidence.get(
                "decompression_total_bytes"
            ),
        }
    return {
        "format": "turbocider-streaming-campaign-verification-v1",
        "schema_version": SCHEMA_VERSION,
        "overall": overall,
        "comparison_kind": comparison_kind,
        "total_blocks": len(blocks),
        "bootstrap_blocks": len(successful_blocks),
        "total_pairs": len(pairs),
        "matched_pairs": len(wall_pairs),
        "failed_samples": failures,
        "hard_failure_samples": hard_failure_samples,
        "quality_failures": quality_failures,
        "quality_not_measured": quality_not_measured,
        "quality_status": quality.get("status", "missing"),
        "environment_status": environment.get("status", "missing"),
        "environment_complete": environment_complete,
        "campaign_status": manifest.get("status", "missing"),
        "campaign_complete": campaign_complete,
        "audit_status": audit_status,
        "audit_passed": audit_status == "passed",
        "audit_observations": audit_observations,
        "source_provenance_complete": provenance_ok,
        "protocol_complete": protocol_ok,
        "memory_evidence": memory_evidence,
        "memory_qualification": (
            memory_evidence.get("qualification")
            if memory_evidence is not None else "NOT_REQUESTED"
        ),
        "p3_environment": p3_environment,
        "p3_result": p3_result,
        "sample_size_sufficient": sample_sufficient,
        "minimum_matched_pairs": minimum_pairs,
        "minimum_tail_requests_per_variant": tail_minimum,
        "required_matched_pairs": required_pairs,
        "completion": {
            "successful_requests": successful_requests,
            "total_requests": total_requests,
            "rate": successful_requests / total_requests,
        },
        "decisions": decisions,
        "metrics": metrics,
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "unit": "abba_block",
        },
        "build_identity": build,
        "policy_sha256": policy_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.bundle.resolve())
    except EvidenceError as exc:
        print(json.dumps({"overall": "INVALID", "error": str(exc)}, indent=2))
        return 2
    output = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
