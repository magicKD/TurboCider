#!/usr/bin/env python3
"""CPU-only tests for deterministic P0-P3 release policy preparation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
sys.path.insert(0, str(ROOT / "tests/native"))

import build_streaming_catalog as builder  # noqa: E402
import prepare_streaming_release_policies as preparer  # noqa: E402
from test_streaming_campaign_verifier import policy as campaign_policy  # noqa: E402
from test_streaming_catalog_builder import record as catalog_record  # noqa: E402


class ReleasePolicyPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tc-policy-set-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record = catalog_record()
        self.record_path = self.root / "record.json"
        self.record_path.write_text(json.dumps({"record": self.record}, indent=2))
        self.templates: dict[str, Path] = {}
        for kind in preparer.KINDS:
            value = campaign_policy(blocks=10)
            value["comparison_kind"] = kind
            value.pop("catalog_binding", None)
            if kind != "P1":
                value.pop("semantic_equivalence", None)
            if kind == "P0":
                value["thresholds"] = {
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
            if kind in ("P2", "P3"):
                value["protocol"]["restart_workers_between_blocks"] = True
                value["memory_sampling"] = {
                    "enabled": True,
                    "interval_ms": 20,
                    "max_gap_ms": 100,
                    "required_variants": ["candidate"],
                }
                for variant in ("baseline", "candidate"):
                    value["variants"][variant]["source_identity"] = {
                        "commit": "a" * 40,
                        "source_manifest_sha256": "b" * 64,
                        "clean": True,
                    }
            if kind == "P3":
                value["memory_sampling"].update({
                    "required_variants": ["baseline", "candidate"],
                    "allow_swap_out": True,
                })
                value["swap_comparison"] = {
                    "revision": "tc-p3-natural-swap-v1",
                    "pressure_source": "natural_low_memory_device",
                    "baseline_role": "resident_or_default",
                    "candidate_role": "public_streaming_exact",
                    "minimum_baseline_swap_runs": 1,
                    "candidate_swap_out_total_ratio_max": 1.0,
                    "reporting_mode": "tradeoff_or_speedup",
                }
            path = self.root / f"{kind}.json"
            path.write_text(json.dumps(value, indent=2))
            self.templates[kind] = path

    def test_prepares_deterministic_exact_binding(self):
        output = self.root / "prepared"
        manifest = preparer.prepare_policy_set(
            self.record_path, self.templates, 8 << 30, output
        )
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["target_bytes"], 8 << 30)
        binding = builder.catalog_binding(self.record)
        for kind in preparer.KINDS:
            policy = json.loads(
                (output / f"{kind.lower()}-campaign-policy.json").read_text()
            )
            self.assertEqual(policy["catalog_binding"], binding)
        p2 = json.loads((output / "p2-campaign-policy.json").read_text())
        self.assertEqual(p2["memory_sampling"]["target_bytes"], 8 << 30)
        self.assertEqual(
            p2["memory_sampling"]["headroom_policy_revision"],
            "tc-public-headroom-v1",
        )

    def test_rejects_template_with_preexisting_binding(self):
        p1 = json.loads(self.templates["P1"].read_text())
        p1["catalog_binding"] = {"stale": True}
        self.templates["P1"].write_text(json.dumps(p1))
        with self.assertRaisesRegex(
            preparer.PolicyPreparationError, "already contains"
        ):
            preparer.prepare_policy_set(
                self.record_path, self.templates, 8 << 30,
                self.root / "prepared",
            )

    def test_rejects_non_public_target(self):
        with self.assertRaisesRegex(
            preparer.PolicyPreparationError, "8/10/12/16/20"
        ):
            preparer.prepare_policy_set(
                self.record_path, self.templates, 6 << 30,
                self.root / "prepared",
            )

    def test_policies_change_when_record_identity_changes(self):
        first_output = self.root / "first"
        first = preparer.prepare_policy_set(
            self.record_path, self.templates, 8 << 30, first_output
        )
        changed = copy.deepcopy(self.record)
        changed["plan"]["layout_digest"] = "9" * 64
        changed_path = self.root / "changed-record.json"
        changed_path.write_text(json.dumps({"record": changed}, indent=2))
        second = preparer.prepare_policy_set(
            changed_path, self.templates, 8 << 30, self.root / "second"
        )
        self.assertNotEqual(
            first["catalog_binding_sha256"],
            second["catalog_binding_sha256"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
