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


SCHEMA_VERSION = 1
VARIANTS = ("baseline", "candidate")
VALID_BLOCK_SEQUENCES = {
    ("baseline", "candidate", "candidate", "baseline"),
    ("candidate", "baseline", "baseline", "candidate"),
}
KIND_TO_THRESHOLD = {"P0": "P0_legacy", "P1": "P1_same_layout"}
SPEC_MAXIMUMS = {
    "P0": {
        "wall_median_ratio_max": 1.02,
        "wall_p95_ratio_max": 1.05,
        "denoise_median_ratio_max": 1.02,
    },
    "P1": {
        "wall_median_ratio_max": 1.03,
        "wall_p95_ratio_max": 1.05,
        "denoise_median_ratio_max": 1.03,
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
    for name in required:
        record = files.get(name)
        if not isinstance(record, dict) or not record.get("sha256"):
            raise EvidenceError(f"manifest lacks hash for {name}")
        path = bundle / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
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
            })
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
    ) if comparison_kind == "P0" else True
    if comparison_kind == "P1":
        semantic = read_json(bundle / "semantic-equivalence.json")
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
                    left = baseline.get("actual_layout")
                    right = candidate.get("actual_layout")
                    if not left or left != right:
                        raise EvidenceError(
                            f"P1 pair {pair_id} lacks an identical actual layout"
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
    prerequisites_complete = (
        audit_status == "passed" and quality_complete and provenance_ok and
        environment_complete and campaign_complete and protocol_ok and
        sample_sufficient and not failures and
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
        hard_failure_samples or quality_failures or audit_status == "failed"
    )
    unsupported_kind = comparison_kind not in KIND_TO_THRESHOLD
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
