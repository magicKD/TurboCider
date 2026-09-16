#!/usr/bin/env python3
"""CPU-only tests for the streaming campaign runner and verifier."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

from run_streaming_campaign import CampaignError, run_campaign  # noqa: E402
from run_streaming_campaign import default_audit  # noqa: E402
from verify_streaming_campaign import EvidenceError, verify  # noqa: E402


def policy(*, blocks: int = 10, fail_runs: list[str] | None = None) -> dict:
    return {
        "schema_version": 1,
        "status": "frozen",
        "comparison_kind": "P1",
        "initial_matched_pairs": blocks * 2,
        "minimum_tail_requests_per_variant": 0,
        "bootstrap_iterations": 1200,
        "bootstrap_seed": 17,
        "thresholds": {
            "P1_same_layout": {
                "wall_median_ratio_max": 1.03,
                "wall_p95_ratio_max": 1.05,
                "denoise_median_ratio_max": 1.03,
                "steady_framework_allocations": 0,
                "steady_framework_thread_creates": 0,
            }
        },
        "workload": {
            "request": {"seed": 42, "output": "${OUTPUT}"},
            "seed_path": "seed",
            "seed_stride": 1,
        },
        "protocol": {
            "measured_blocks": blocks,
            "warmup_requests_per_variant": 1,
            "request_timeout_seconds": 5,
            "worker_start_timeout_seconds": 5,
            "cache_condition": "synthetic-fixed",
            "stopping_rule": "fixed-block-count",
            "quantile_estimator": "linear-interpolated-p95",
            "bootstrap_unit": "abba_block",
            "multiplicity_policy": "single-workload-confirmation",
            "exclusion_policy": "no-exclusions-after-freeze",
            "timing_scope": "request-wall-and-denoise",
            "launch_pressure": False,
        },
        "quality": {
            "mode": "artifact_sha256_equal",
            "artifacts": [{"name": "latent", "path": "${OUTPUT}"}],
        },
        "semantic_equivalence": {
            "declared_equivalent": True,
            "fields": ["layout", "seed", "artifact"],
        },
        "variants": {
            "baseline": {
                "backend": "synthetic",
                "synthetic": {
                    "request_wall_seconds": 100.0,
                    "denoise_seconds": 80.0,
                    "layout_digest": "same-layout",
                    "artifact_payloads": {"latent": "same-latent"},
                    "fail_runs": fail_runs or [],
                },
            },
            "candidate": {
                "backend": "synthetic",
                "synthetic": {
                    "request_wall_seconds": 101.0,
                    "denoise_seconds": 80.5,
                    "layout_digest": "same-layout",
                    "artifact_payloads": {"latent": "same-latent"},
                    "fail_runs": [],
                },
            },
        },
    }


def passed_audit() -> dict:
    return {
        "format": "turbocider-streaming-audit-v1",
        "status": "passed",
        "passed": True,
        "new_framework_hooks": 0,
        "new_memory_probes": 0,
        "new_worker_threads": 0,
        "new_pool_allocations": 0,
        "new_cache_clear_or_unload_calls": 0,
        "steady_framework_allocations": 0,
        "steady_framework_thread_creates": 0,
    }


class CampaignTests(unittest.TestCase):
    def test_default_audit_requires_independent_audit_build(self):
        rows = [{
            "variant": "candidate", "status": "success",
            "run_id": "candidate-0",
            "runtime_audit": {
                "audit_available": False,
                "block_streaming_enabled": False,
            },
        }]
        result = default_audit(rows, {"comparison_kind": "P0"})
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["passed"])

    def test_default_audit_aggregates_audit_build_counters(self):
        rows = []
        for index in range(2):
            rows.append({
                "variant": "candidate", "status": "success",
                "run_id": f"candidate-{index}",
                "runtime_audit": {
                    "audit_available": True,
                    "block_streaming_enabled": False,
                    "new_framework_hooks": 0,
                    "new_memory_probes": 0,
                    "new_worker_threads": 0,
                    "new_pool_allocations": 0,
                    "new_cache_clear_or_unload_calls": 0,
                },
            })
        result = default_audit(rows, {"comparison_kind": "P0"})
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertEqual(result["runtime_observations"]["counter_totals"], {
            "new_framework_hooks": 0,
            "new_memory_probes": 0,
            "new_worker_threads": 0,
            "new_pool_allocations": 0,
            "new_cache_clear_or_unload_calls": 0,
        })

    def run_bundle(self, campaign: dict, audit: dict | None = None) -> Path:
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-test-"))
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(campaign))
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit or passed_audit()))
        environment_path = root / "environment.json"
        environment_path.write_text(json.dumps({
            "format": "turbocider-streaming-environment-v1",
            "status": "complete",
            "gpu": "synthetic",
            "ram_bytes": 64 << 30,
            "ssd": "synthetic",
            "power": "fixed",
            "thermal": "fixed",
            "pressure": "none",
        }))
        summary = run_campaign(
            policy_path, root / "bundle", audit_path, environment_path,
        )
        self.assertTrue((root / "bundle" / "raw-samples.jsonl").is_file())
        self.assertTrue((root / "bundle" / "summary.json").is_file())
        return root / "bundle"

    def test_synthetic_abba_campaign_passes_with_persistent_workers(self):
        bundle = self.run_bundle(policy())
        result = verify(bundle)
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(result["bootstrap"]["unit"], "abba_block")
        self.assertEqual(result["total_blocks"], 10)
        self.assertEqual(result["matched_pairs"], 20)
        self.assertEqual(result["decisions"]["wall_median"], "PASS")

        rows = [
            json.loads(line)
            for line in (bundle / "raw-samples.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 40)
        self.assertEqual(
            {row["worker_pid"] for row in rows},
            {rows[0]["worker_pid"], rows[1]["worker_pid"]},
        )
        self.assertNotEqual(rows[0]["worker_pid"], rows[1]["worker_pid"])
        for variant in ("baseline", "candidate"):
            self.assertEqual(
                len({row["worker_pid"] for row in rows
                     if row["variant"] == variant}),
                1,
            )
        self.assertEqual(
            [row["variant"] for row in rows[:4]],
            ["baseline", "candidate", "candidate", "baseline"],
        )

    def test_partial_audit_is_inconclusive_not_pass(self):
        bundle = self.run_bundle(policy(), audit={
            "format": "turbocider-streaming-audit-v1",
            "status": "partial",
            "passed": None,
        })
        result = verify(bundle)
        self.assertEqual(result["overall"], "INCONCLUSIVE")
        self.assertEqual(result["audit_status"], "partial")
        self.assertNotEqual(result["decisions"]["wall_median"], "PASS")

    def test_failure_is_preserved_and_blocks_pass(self):
        campaign = policy(blocks=2, fail_runs=["block-000-position-0-baseline"])
        bundle = self.run_bundle(campaign)
        rows = [
            json.loads(line)
            for line in (bundle / "raw-samples.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any(row["status"] == "failure" for row in rows))
        result = verify(bundle)
        self.assertEqual(result["overall"], "FAIL")
        self.assertTrue(result["failed_samples"])

    def test_timeout_aborts_but_preserves_full_planned_raw_sequence(self):
        campaign = policy(blocks=1)
        campaign["comparison_kind"] = "P0"
        campaign["thresholds"] = {
            "P0_legacy": {
                "wall_median_ratio_max": 1.02,
                "wall_p95_ratio_max": 1.05,
                "denoise_median_ratio_max": 1.02,
                "new_framework_hooks": 0,
                "new_memory_probes": 0,
                "new_worker_threads": 0,
                "new_pool_allocations": 0,
                "new_cache_clear_or_unload_calls": 0,
            }
        }
        campaign["protocol"]["warmup_requests_per_variant"] = 0
        campaign["protocol"]["request_timeout_seconds"] = 0.02
        campaign["variants"]["candidate"]["synthetic"]["sleep_seconds"] = 0.2
        bundle = self.run_bundle(campaign)
        rows = [
            json.loads(line)
            for line in (bundle / "raw-samples.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 4)
        self.assertIn("timeout", {row["status"] for row in rows})
        self.assertIn("not_run_after_abort", {row["status"] for row in rows})
        result = verify(bundle)
        self.assertEqual(result["overall"], "FAIL")
        self.assertTrue(result["hard_failure_samples"])

    def test_p0_without_source_manifest_cannot_pass(self):
        campaign = policy(blocks=1)
        campaign["comparison_kind"] = "P0"
        campaign["thresholds"] = {
            "P0_legacy": {
                "wall_median_ratio_max": 1.02,
                "wall_p95_ratio_max": 1.05,
                "denoise_median_ratio_max": 1.02,
                "new_framework_hooks": 0,
                "new_memory_probes": 0,
                "new_worker_threads": 0,
                "new_pool_allocations": 0,
                "new_cache_clear_or_unload_calls": 0,
            }
        }
        bundle = self.run_bundle(campaign)
        result = verify(bundle)
        self.assertEqual(result["overall"], "INCONCLUSIVE")
        self.assertFalse(result["source_provenance_complete"])

    def test_malformed_block_is_rejected(self):
        bundle = self.run_bundle(policy(blocks=2))
        path = bundle / "raw-samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["position"] = 3
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["files"]["raw-samples.jsonl"]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        manifest["files"]["raw-samples.jsonl"]["bytes"] = path.stat().st_size
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(EvidenceError):
            verify(bundle)

    def test_policy_must_not_launch_pressure(self):
        campaign = policy(blocks=1)
        campaign["protocol"]["launch_pressure"] = True
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-policy-"))
        path = root / "policy.json"
        path.write_text(json.dumps(campaign))
        with self.assertRaises(CampaignError):
            run_campaign(path, root / "bundle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
