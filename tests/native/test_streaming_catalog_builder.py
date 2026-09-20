#!/usr/bin/env python3
"""CPU-only contract tests for the immutable streaming catalog builder."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

import build_streaming_catalog as builder  # noqa: E402


GIB = 1 << 30


def record() -> dict:
    return {
        "id": "fixture-id",
        "revision": 1,
        "catalog_revision": "tc-streaming-catalog-test-v1",
        "source": {
            "model_variant": "original-bf16",
            "weight_format": "safetensors-bf16",
            "artifact_manifest_digest": "a" * 64,
            "source_snapshot_digest": "b" * 64,
        },
        "workload": {
            "model": "z-image-turbo",
            "operation": "image.generate",
            "execution": "gpu",
            "device_class": "apple_metal_gpu",
            "execution_container": "embedded_app",
            "width": 512,
            "height": 512,
            "frames": 1,
            "fps": 0,
            "steps": 9,
            "batch": 1,
            "audio": False,
            "dynamic_text": True,
            "approximation": False,
            "conditioning_revision": "zimage-conditioning-v1",
            "vae_policy_revision": "zimage-vae-v1",
            "feature_digest": "c" * 64,
            "token_shapes": [{
                "encoder": "qwen3",
                "tokenizer_revision": "tokenizer-v1",
                "template_revision": "template-v1",
                "valid_rows": 16,
                "padded_rows": 32,
                "compute_rows": 32,
            }],
        },
        "runtime": {
            "turbocider_build_id": "build-test",
            "runtime_revision": "runtime-v1",
            "adapter_revision": "adapter-v1",
            "reader_revision": "reader-v1",
            "kernel_revision": "kernel-v1",
            "allocator_policy_revision": "allocator-v1",
        },
        "device": {
            "minimum_physical_memory_bytes": 16 * GIB,
            "maximum_physical_memory_bytes": 16 * GIB,
        },
        "plan": {
            "canonical_config": {
                "enabled": True,
                "schema_version": 1,
                "selection": "manual",
                "retention": "request",
                "stages": {
                    "denoiser": {
                        "residency": "streamed",
                        "block_group_size": 1,
                        "slot_count": 2,
                        "resident_prefix_blocks": 14,
                        "prefetch_distance": 0,
                        "io_workers": 1,
                    }
                },
            },
            "layout_digest": "d" * 64,
            "component_policy_revision": "zimage-components-v1",
            "pass_transition": "reload",
            "multi_pool_policy": "serial",
        },
        "calibration": {
            "complete": True,
            "calibrated_request_bytes": 7 * GIB,
            "scope": builder.CALIBRATION_SCOPE,
            "estimator_revision": builder.CALIBRATION_ESTIMATOR,
            "calibration_id": "calibration-test-v1",
            "execution_container": "embedded_app",
            "evidence_digest": "e" * 64,
            "confirmation_sample_count": 20,
            "maximum_sample_gap_ns": 20_000_000,
        },
        "performance": {
            "rank": 1,
            "logical_read_bytes": 1234,
            "profile_id": "performance-test-v1",
            "comparison_kind": "P1_same_layout",
            "confidence_status": "PASS",
            "evidence_digest": "f" * 64,
        },
        "release": {
            "channel": "staging",
            "revoked": False,
            "reviewed_commit": "1" * 40,
            "review_digest": "2" * 64,
        },
    }


def native_fixture_record() -> dict:
    value = record()
    value["id"] = "canonical-fixture"
    value["catalog_revision"] = "test-r1"
    value["workload"]["device_class"] = "Apple Test GPU/16GiB"
    value["calibration"]["estimator_revision"] = "observed-tree-max-v1"
    value["calibration"]["confirmation_sample_count"] = 13
    value["performance"].update({
        "rank": 2,
        "logical_read_bytes": 100,
        "comparison_kind": "strategy_tradeoff",
        "confidence_status": "pass",
    })
    value["release"].update({
        "channel": "public-experimental",
        "reviewed_commit": "test-review-commit",
        "review_digest": "1" * 64,
    })
    return value


class CatalogBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tc-catalog-builder-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_fixture(self, *, channel: str = "staging") -> dict:
        fixture_root = Path(tempfile.mkdtemp(prefix="fixture-", dir=self.root))
        value = record()
        value["release"]["channel"] = channel
        commit = value["release"]["reviewed_commit"]
        summaries: dict[str, dict] = {}
        bundle_paths: dict[str, Path] = {}
        kinds = ("P0", "P1", "P2", "P3")
        for index, kind in enumerate(kinds):
            bundle = fixture_root / kind
            bundle.mkdir()
            bundle_paths[kind] = bundle
            memory = None
            if kind == "P2":
                target = 8 * GIB
                memory = {
                    "qualification": "PASS",
                    "over_target": [],
                    "unexpected_swap": [],
                    "incomplete": [],
                    "insufficient_variants": [],
                    "fresh_process_failures": [],
                    "fresh_process_generations": {"candidate": 10},
                    "required_count": 20,
                    "required_variants": ["candidate"],
                    "target_bytes": target,
                    "allowed_peak_bytes": target - max(
                        512 << 20, (target * 10 + 99) // 100
                    ),
                    "maximum_sample_gap_ns": 20_000_000,
                    "allowed_max_gap_ns": 100_000_000,
                    "peak_p95_bytes": {"candidate": 7 * GIB},
                }
            elif kind == "P3":
                memory = {
                    "qualification": "PASS",
                    "required_count": 40,
                    "required_count_by_variant": {
                        "baseline": 20, "candidate": 20,
                    },
                    "required_variants": ["baseline", "candidate"],
                    "swap_out_total_bytes": {
                        "baseline": 10_000, "candidate": 1_000,
                    },
                    "swap_in_total_bytes": {
                        "baseline": 20_000, "candidate": 2_000,
                    },
                    "compression_total_bytes": {
                        "baseline": 30_000, "candidate": 3_000,
                    },
                    "decompression_total_bytes": {
                        "baseline": 40_000, "candidate": 4_000,
                    },
                    "candidate_swap_out_total_ratio": 0.1,
                    "p3_swap_status": "PASS",
                }
            summaries[kind] = {
                "format": "turbocider-streaming-campaign-verification-v1",
                "schema_version": 1,
                "overall": "PASS",
                "comparison_kind": kind,
                "memory_evidence": memory,
                "p3_result": (
                    {
                        "qualification": "PASS",
                        "classification": "lower_swap_tradeoff",
                        "swap_status": "PASS",
                        "candidate_swap_out_total_ratio": 0.1,
                    }
                    if kind == "P3" else None
                ),
                "build_identity": {
                    "candidate": {
                        "source_identity": {
                            "commit": commit,
                            "source_manifest_sha256": chr(ord("a") + index) * 64,
                            "clean": True,
                        }
                    }
                },
                "policy_sha256": chr(ord("1") + index) * 64,
            }
            (bundle / "summary.json").write_text(
                json.dumps(summaries[kind], indent=2) + "\n"
            )

        digests = {
            kind: builder.sha256_file(bundle_paths[kind] / "summary.json")
            for kind in kinds
        }
        value["calibration"]["evidence_digest"] = digests["P2"]
        value["performance"]["evidence_digest"] = digests["P1"]
        for kind in kinds:
            policy = {
                "comparison_kind": kind,
                "catalog_binding": builder.catalog_binding(value),
            }
            if kind == "P2":
                policy["memory_sampling"] = {
                    "enabled": True,
                    "required_variants": ["candidate"],
                    "target_bytes": 8 * GIB,
                    "headroom_policy_revision": builder.HEADROOM_REVISION,
                }
            if kind == "P3":
                policy["memory_sampling"] = {
                    "enabled": True,
                    "required_variants": ["baseline", "candidate"],
                    "allow_swap_out": True,
                }
                policy["swap_comparison"] = {
                    "revision": "tc-p3-natural-swap-v1",
                    "pressure_source": "externally_managed_fixed_pressure",
                    "baseline_role": "resident_or_default",
                    "candidate_role": "public_streaming_exact",
                    "minimum_baseline_swap_runs": 1,
                    "candidate_swap_out_total_ratio_max": 1.0,
                    "reporting_mode": "tradeoff_or_speedup",
                }
            (bundle_paths[kind] / "campaign-policy.json").write_text(
                json.dumps(policy, indent=2) + "\n"
            )
            rows = []
            if kind != "P0":
                rows = [{
                    "variant": "candidate",
                    "status": "success",
                    "actual_layout_digest": value["plan"]["layout_digest"],
                }]
            (bundle_paths[kind] / "raw-samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

        required = builder.PUBLIC_GATES if channel.startswith("public-") else builder.STAGING_GATES
        identity = builder.record_identity_digest(value)
        review = {
            "schema": builder.REVIEW_SCHEMA,
            "status": "approved",
            "reviewed_commit": commit,
            "record_identity_digest": identity,
            "evidence_summary_sha256": {
                kind: digests[kind] for kind in required
            },
            "reviewers": {
                "runtime": "runtime-owner",
                "model": "model-owner",
                "performance": "performance-owner",
                "release": "release-owner",
            },
            "review_digest": "0" * 64,
        }
        review["review_digest"] = builder._review_digest(review)
        value["release"]["review_digest"] = review["review_digest"]
        record_path = fixture_root / "record.json"
        review_path = fixture_root / "review.json"
        record_path.write_text(json.dumps({"record": value}, indent=2) + "\n")
        review_path.write_text(json.dumps(review, indent=2) + "\n")
        return {
            "record": value,
            "record_path": record_path,
            "review": review,
            "review_path": review_path,
            "bundles": bundle_paths,
            "summaries": summaries,
            "digests": digests,
        }

    def build(self, fixture: dict, **kwargs) -> dict:
        bundles = fixture["bundles"]

        def verified(path: Path) -> dict:
            return json.loads((Path(path) / "summary.json").read_text())

        with mock.patch.object(builder, "verify_campaign", side_effect=verified):
            return builder.build_record(
                bundles["P2"], fixture["record_path"], fixture["review_path"],
                bundles["P1"], bundles["P0"], kwargs.get("swap_bundle"),
                kwargs.get("existing_catalog"),
            )

    def test_native_digest_wraps_canonical_record_bytes(self):
        value = record()
        builder.validate_record_shape(value)
        raw = builder.canonical_record_bytes(value)
        wrapped = builder.CanonicalEncoder(builder.RECORD_DIGEST_SCHEMA)
        wrapped.string_field("canonical_record", raw)
        self.assertEqual(builder.canonical_record_digest(value), wrapped.digest())
        self.assertNotEqual(builder.canonical_record_digest(value), hashlib.sha256(raw).hexdigest())

    def test_python_digest_matches_native_locked_fixture(self):
        self.assertEqual(
            builder.canonical_record_digest(native_fixture_record()),
            "13b5797176d713924b9c16857acbf0f7a35313d2426a22fdca38c488b3ea3df1",
        )

    def test_native_calibration_field_order_is_explicit(self):
        value = record()
        digest_before = builder.canonical_record_digest(value)
        value["calibration"]["confirmation_sample_count"] += 1
        digest_after = builder.canonical_record_digest(value)
        self.assertNotEqual(digest_before, digest_after)

    def test_portable_v2_native_digest_and_strict_identity_schema(self):
        value = native_fixture_record()
        value["source"]["identity_version"] = 2
        del value["source"]["source_snapshot_digest"]
        self.assertEqual(builder.canonical_record_digest(value),
                         "c322e18cabc2ed4756a8cbbc468c24989ba175dda64745902014a1971aca0bb0")
        value = record()
        value["source"]["identity_version"] = 2
        del value["source"]["source_snapshot_digest"]
        builder.validate_record_shape(value)
        for version in (0, 3, True, "2"):
            invalid = copy.deepcopy(value)
            invalid["source"]["identity_version"] = version
            with self.assertRaises(builder.CatalogBuildError):
                builder.validate_record_shape(invalid)
            with self.assertRaises(builder.CatalogBuildError):
                builder.canonical_record_digest(invalid)
        invalid = copy.deepcopy(value)
        invalid["source"]["source_snapshot_digest"] = "b" * 64
        with self.assertRaises(builder.CatalogBuildError):
            builder.validate_record_shape(invalid)
        with self.assertRaises(builder.CatalogBuildError):
            builder.canonical_record_digest(invalid)
        legacy = record()
        explicit = copy.deepcopy(legacy)
        explicit["source"]["identity_version"] = 1
        builder.validate_record_shape(explicit)
        self.assertEqual(builder.canonical_record_digest(legacy),
                         builder.canonical_record_digest(explicit))

    def test_record_shape_rejects_unencoded_fields(self):
        value = record()
        value["performance"]["median_ratio"] = 1.0
        with self.assertRaisesRegex(builder.CatalogBuildError, "unencoded fields"):
            builder.validate_record_shape(value)

    def test_record_shape_rejects_invalid_slot_policy(self):
        value = record()
        value["plan"]["canonical_config"]["stages"]["denoiser"]["prefetch_distance"] = 2
        with self.assertRaisesRegex(builder.CatalogBuildError, "slot policy"):
            builder.validate_record_shape(value)

    def test_p2_requires_candidate_only_and_twenty_samples(self):
        summary = {
            "overall": "PASS",
            "comparison_kind": "P2",
            "memory_evidence": {
                "qualification": "PASS",
                "over_target": [],
                "unexpected_swap": [],
                "incomplete": [],
                "insufficient_variants": [],
                "fresh_process_failures": [],
                "required_count": 19,
                "required_variants": ["candidate"],
                "target_bytes": 8 * GIB,
                "allowed_peak_bytes": 7 * GIB,
                "maximum_sample_gap_ns": 1,
                "allowed_max_gap_ns": 2,
                "peak_p95_bytes": {"candidate": 6 * GIB},
            },
        }
        policy = {
            "memory_sampling": {
                "target_bytes": 8 * GIB,
                "headroom_policy_revision": builder.HEADROOM_REVISION,
            }
        }
        with tempfile.TemporaryDirectory(prefix="tc-builder-p2-") as raw:
            path = Path(raw) / "summary.json"
            path.write_text("{}")
            with self.assertRaisesRegex(builder.CatalogBuildError, "20 runs"):
                builder._validate_p2_summary(Path(raw), summary, policy)

    def test_builder_requires_all_staging_evidence(self):
        value = record()
        with tempfile.TemporaryDirectory(prefix="tc-builder-input-") as raw:
            root = Path(raw)
            input_path = root / "record.json"
            input_path.write_text(json.dumps(value))
            with self.assertRaisesRegex(builder.CatalogBuildError, "P0 default"):
                builder.build_record(root / "p2", input_path, None)

    def test_verified_staging_record_is_deterministic(self):
        fixture = self.make_fixture()
        first = self.build(fixture)
        second = self.build(fixture)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "verified")
        self.assertTrue(first["record"]["id"].startswith("tc-streaming-"))
        self.assertEqual(
            first["record"]["canonical_record_digest"],
            builder.canonical_record_digest(first["record"]),
        )
        self.assertEqual(first["evidence"]["target_bytes"], 8 * GIB)

    def test_inconclusive_campaign_is_rejected(self):
        fixture = self.make_fixture()
        path = fixture["bundles"]["P2"] / "summary.json"
        summary = json.loads(path.read_text())
        summary["overall"] = "INCONCLUSIVE"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        with self.assertRaisesRegex(builder.CatalogBuildError, "not PASS"):
            self.build(fixture)

    def test_peak_or_swap_failure_is_rejected(self):
        for field in ("over_target", "unexpected_swap"):
            with self.subTest(field=field):
                fixture = self.make_fixture()
                path = fixture["bundles"]["P2"] / "summary.json"
                summary = json.loads(path.read_text())
                summary["memory_evidence"][field] = ["candidate-0"]
                path.write_text(json.dumps(summary, indent=2) + "\n")
                with self.assertRaisesRegex(builder.CatalogBuildError, field):
                    self.build(fixture)

    def test_review_digest_mismatch_is_rejected(self):
        fixture = self.make_fixture()
        review = copy.deepcopy(fixture["review"])
        review["reviewers"]["release"] = "different-owner"
        fixture["review_path"].write_text(json.dumps(review, indent=2) + "\n")
        with self.assertRaisesRegex(builder.CatalogBuildError, "review digest"):
            self.build(fixture)

    def test_layout_digest_mismatch_is_rejected(self):
        fixture = self.make_fixture()
        path = fixture["bundles"]["P1"] / "raw-samples.jsonl"
        path.write_text(json.dumps({
            "variant": "candidate",
            "status": "success",
            "actual_layout_digest": "9" * 64,
        }) + "\n")
        with self.assertRaisesRegex(builder.CatalogBuildError, "P1 candidate receipt"):
            self.build(fixture)

    def test_existing_duplicate_identity_is_rejected(self):
        fixture = self.make_fixture()
        built = self.build(fixture)
        existing = self.root / "existing.json"
        existing.write_text(json.dumps(built, indent=2) + "\n")
        with self.assertRaisesRegex(builder.CatalogBuildError, "record id"):
            self.build(fixture, existing_catalog=existing)

    def test_public_record_requires_p3(self):
        fixture = self.make_fixture(channel="public-experimental")
        with self.assertRaisesRegex(builder.CatalogBuildError, "P3 swap"):
            self.build(fixture)

    def test_verified_public_record_accepts_p3_tradeoff(self):
        fixture = self.make_fixture(channel="public-experimental")
        built = self.build(fixture, swap_bundle=fixture["bundles"]["P3"])
        self.assertEqual(built["status"], "verified")
        self.assertEqual(
            built["evidence"]["p3"]["classification"],
            "lower_swap_tradeoff",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
