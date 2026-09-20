#!/usr/bin/env python3
"""CPU-only tests for the streaming campaign runner and verifier."""

from __future__ import annotations

import json
import hashlib
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

import run_streaming_campaign as campaign_runner  # noqa: E402
from run_streaming_campaign import (  # noqa: E402
    CampaignError,
    actual_semantic_layout,
    build_identity,
    create_native_engine,
    default_audit,
    hash_artifacts,
    receive_message,
    request_semantic_identity,
    run_campaign,
    send_message,
    worker_main,
)
from verify_streaming_campaign import EvidenceError, verify  # noqa: E402


def policy(*, blocks: int = 10, fail_runs: list[str] | None = None) -> dict:
    semantic_layout = {
        "stage": "denoiser",
        "resident_prefix_blocks": 8,
        "block_group_size": 1,
        "slot_count": 3,
        "prefetch_distance": 2,
        "io_workers": 3,
        "group_count": 40,
        "pass_count": 11,
        "startup_policy": "prefetch_window_before_prefix",
        "pass_transition": "reload",
        "retention": "request",
        "reader_revision": 1,
        "weight_format": "convrot-int8-g256",
        "kernel_revision": "synthetic-kernel-v1",
        "conditioning_recipe": "scalar-conditioning-v1",
        "upsample_boundary": "after-stage1-pool-retained",
        "total_fills": 440,
    }
    return {
        "schema_version": 1,
        "status": "frozen",
        "comparison_kind": "P1",
        "engine_lifecycle": "per_request",
        "initial_matched_pairs": blocks * 2,
        "minimum_tail_requests_per_variant": 0,
        "bootstrap_iterations": 1200,
        "bootstrap_seed": 17,
        "thresholds": {
            "P1_same_layout": {
                "wall_median_ratio_max": 1.02,
                "wall_p95_ratio_max": 1.05,
                "denoise_median_ratio_max": 1.02,
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
            "expected_actual": {
                **semantic_layout,
                "engine_lifecycle": "per_request",
            },
        },
        "variants": {
            "baseline": {
                "backend": "synthetic",
                "synthetic": {
                    "request_wall_seconds": 100.0,
                    "denoise_seconds": 80.0,
                    "layout_digest": "same-layout",
                    "actual_semantic_layout": dict(semantic_layout),
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
                    "actual_semantic_layout": dict(semantic_layout),
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


class ContainerConstructorTests(unittest.TestCase):
    def test_controlled_entry_selection(self):
        self.assertEqual(campaign_runner.native_constructor_name({}), "tc_engine_create_model_worker")
        self.assertEqual(campaign_runner.native_constructor_name({"constructor": "candidate"}),
                         "tc_engine_create_model_candidate_worker")
        self.assertEqual(campaign_runner.native_constructor_name({"execution_container": "embedded_app"}),
                         "tc_engine_create_model")
        with self.assertRaises(CampaignError):
            campaign_runner.native_constructor_name({"execution_container": "arbitrary"})


class CampaignTests(unittest.TestCase):
    def test_quality_artifact_canonicalizer_validation_and_hashing(self):
        campaign = policy(blocks=1)
        campaign["quality"]["artifacts"][0]["canonicalizer"] = (
            "ffmpeg-rgb24-v1"
        )
        campaign_runner.validate_policy(campaign)

        invalid = json.loads(json.dumps(campaign))
        invalid["quality"]["artifacts"][0]["canonicalizer"] = "unknown"
        with self.assertRaisesRegex(CampaignError, "unsupported canonicalizer"):
            campaign_runner.validate_policy(invalid)

        root = Path(tempfile.mkdtemp(prefix="tc-canonical-artifact-"))
        source = root / "video.mp4"
        source.write_bytes(b"container")
        completed = mock.Mock(returncode=0, stdout=b"decoded-rgb", stderr=b"")
        version = mock.Mock(
            returncode=0, stdout="ffmpeg version fixture\n", stderr=""
        )
        executable = root / "ffmpeg"
        executable.write_bytes(b"fixture-ffmpeg")
        with (
            mock.patch.object(
                campaign_runner.shutil, "which", return_value=str(executable)
            ),
            mock.patch.object(
                campaign_runner.subprocess, "run",
                side_effect=[completed, version],
            ),
        ):
            result = hash_artifacts(
                {"video": str(source)},
                {"video": "ffmpeg-rgb24-v1"},
            )["video"]
        self.assertEqual(
            result["sha256"], hashlib.sha256(b"decoded-rgb").hexdigest()
        )
        self.assertEqual(result["source_sha256"], hashlib.sha256(
            b"container"
        ).hexdigest())
        self.assertEqual(result["canonicalizer"], "ffmpeg-rgb24-v1")
        self.assertEqual(result["canonical_bytes"], len(b"decoded-rgb"))

    def test_memory_sampling_stops_before_artifact_canonicalization(self):
        root = Path(tempfile.mkdtemp(prefix="tc-deferred-artifact-hash-"))
        artifact = root / "video.mp4"
        artifact.write_bytes(b"container")
        events: list[str] = []

        class FakeWorker:
            pid = 123

            def run(self, command):
                events.append("worker")
                self.assert_deferred(command)
                return {"status": "success", "artifacts": {}}

            @staticmethod
            def assert_deferred(command):
                if command.get("defer_artifact_hashing") is not True:
                    raise AssertionError("worker hashing was not deferred")

        class FakeSampler:
            def __init__(
                self, worker_pid, evidence_path, correlation_id, config
            ):
                self.evidence_path = evidence_path

            def start(self):
                events.append("sampler_start")
                self.evidence_path.parent.mkdir(parents=True, exist_ok=True)

            def stop(self):
                events.append("sampler_stop")
                return {
                    "schema": "turbocider-streaming-memory-summary-v1",
                    "status": "complete",
                    "complete": True,
                }

        command = {
            "type": "run",
            "run_id": "sample-0",
            "variant": "candidate",
            "artifact_paths": {"video": str(artifact)},
            "artifact_canonicalizers": {"video": "ffmpeg-rgb24-v1"},
        }
        campaign = {
            "memory_sampling": {
                "enabled": True,
                "include_warmups": True,
            }
        }

        def deferred_hash(paths, canonicalizers):
            events.append("hash")
            return {"video": {"sha256": "canonical"}}

        with (
            mock.patch.object(
                campaign_runner, "RequestMemorySampler", FakeSampler
            ),
            mock.patch.object(
                campaign_runner, "hash_artifacts", side_effect=deferred_hash
            ),
        ):
            response = campaign_runner.run_worker_request(
                FakeWorker(), command, root, campaign, "measured"
            )

        self.assertEqual(
            events, ["sampler_start", "worker", "sampler_stop", "hash"]
        )
        self.assertEqual(
            response["artifacts"], {"video": {"sha256": "canonical"}}
        )
        self.assertNotIn("defer_artifact_hashing", command)

    def test_native_test_catalog_is_public_only_and_hashed(self):
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-test-catalog-"))
        library = root / "fixture.dylib"
        library.write_bytes(b"fixture-library")
        catalog = root / "catalog.json"
        catalog.write_text('{"fixture":true}\n')
        model = root / "model"
        model.mkdir()
        campaign = policy(blocks=1)
        for variant in ("baseline", "candidate"):
            campaign["variants"][variant] = {
                "backend": "native",
                "library": str(library),
                "model_id": "fixture-model",
                "model_path": str(model),
                "constructor": "public",
                "test_streaming_catalog": str(catalog),
            }
        campaign_runner.validate_policy(campaign)
        identity = build_identity(campaign)
        expected = hashlib.sha256(catalog.read_bytes()).hexdigest()
        for variant in ("baseline", "candidate"):
            self.assertEqual(
                identity[variant]["test_streaming_catalog_sha256"], expected
            )
            self.assertEqual(
                identity[variant]["test_streaming_catalog_size_bytes"],
                catalog.stat().st_size,
            )

        invalid = json.loads(json.dumps(campaign))
        invalid["variants"]["candidate"]["constructor"] = "candidate"
        with self.assertRaisesRegex(CampaignError, "public constructor"):
            campaign_runner.validate_policy(invalid)

        invalid = json.loads(json.dumps(campaign))
        invalid["variants"]["candidate"]["test_streaming_catalog"] = str(
            root / "missing.json"
        )
        with self.assertRaisesRegex(CampaignError, "catalog is missing"):
            campaign_runner.validate_policy(invalid)

    def test_create_native_engine_installs_test_catalog_before_use(self):
        root = Path(tempfile.mkdtemp(prefix="tc-engine-test-catalog-"))
        catalog = root / "catalog.json"
        catalog.write_bytes(b'{"schema":"fixture"}')
        calls: list[tuple[int, bytes]] = []
        freed: list[int] = []

        class FakeLibrary:
            _tc_engine_test_set_streaming_catalog = None

            @staticmethod
            def tc_engine_create_model_worker(_model, _path, engine, _error):
                engine._obj.value = 123
                return 0

            @staticmethod
            def tc_engine_free(engine):
                freed.append(int(engine.value))

        library = FakeLibrary()

        def install(engine, payload, _error):
            calls.append((int(engine.value), payload))
            return 0

        library._tc_engine_test_set_streaming_catalog = install
        engine = create_native_engine(library, {
            "model_id": "fixture-model",
            "model_path": str(root),
            "constructor": "public",
            "test_streaming_catalog": str(catalog),
        })
        self.assertEqual(engine.value, 123)
        self.assertEqual(calls, [(123, catalog.read_bytes())])
        self.assertEqual(freed, [])
        config = {"model_id": "fixture-model", "model_path": str(root),
                  "verify_streaming_sources": True,
                  "test_streaming_catalog": str(catalog)}
        proof = {"scope": "engine_setup", "report": {"status": "verified"}}
        def verify(lib, value):
            self.assertEqual(len(calls), 1)  # No second catalog installation yet.
            self.assertEqual(value.value, 123)
            return proof
        with mock.patch.object(campaign_runner, "verify_native_sources", side_effect=verify):
            create_native_engine(library, config)
        self.assertEqual(library._tc_source_verification, proof)
        self.assertEqual(len(calls), 2)
        with mock.patch.object(campaign_runner, "verify_native_sources", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(CampaignError, "cancelled"):
                create_native_engine(library, config)
        self.assertEqual(freed, [123])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(library._tc_source_verification)
        library._tc_source_verification = proof
        library.tc_engine_create_model_worker = lambda *_: 1
        with self.assertRaisesRegex(CampaignError, "native engine creation failed"):
            create_native_engine(library, config)
        self.assertIsNone(library._tc_source_verification)
        self.assertEqual(freed, [123])

    def test_probe_backend_reuses_semantic_and_quality_verifier(self):
        root = Path(tempfile.mkdtemp(prefix="tc-probe-campaign-"))
        script = root / "probe.py"
        semantic = policy(blocks=1)["semantic_equivalence"]["expected_actual"]
        actual = {
            key: value for key, value in semantic.items()
            if key not in ("engine_lifecycle", "total_fills")
        }
        script.write_text(
            "import json, pathlib, sys\n"
            "output, pair_id, variant, request_path = sys.argv[1:5]\n"
            "pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)\n"
            "pathlib.Path(output).write_bytes(pair_id.encode())\n"
            "assert pathlib.Path(request_path).is_file()\n"
            "assert json.loads(pathlib.Path(request_path).read_text())['seed'] >= 42\n"
            f"actual = {actual!r}\n"
            "wall = 1.01 if variant == 'candidate' else 1.0\n"
            "denoise = 0.805 if variant == 'candidate' else 0.8\n"
            "print(json.dumps({"
            "'schema':'test-streaming-probe-v1',"
            "'timings_seconds':{'request_wall':wall,'denoise':denoise},"
            "'block_streaming':{'enabled':True,"
            "'request_slot_allocations':3,'request_slot_refills':440,"
            "'request_slot_fills':440,'actual_layout':actual}}))\n"
        )
        campaign = policy(blocks=1)
        campaign["protocol"]["warmup_requests_per_variant"] = 0
        for variant in ("baseline", "candidate"):
            campaign["variants"][variant] = {
                "backend": "probe",
                "executable": sys.executable,
                "model_path": str(root),
                "output_artifact": "latent",
                "arguments": [
                    str(script), "${OUTPUT}", "${PAIR_ID}", variant,
                    "${REQUEST}",
                ],
                "result_schema": "test-streaming-probe-v1",
            }
        bundle = self.run_bundle(campaign)
        rows = [
            json.loads(line)
            for line in (bundle / "raw-samples.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["status"] == "success" for row in rows))
        self.assertTrue(all(
            not row["actual_semantic_layout_missing_fields"] for row in rows
        ))
        quality = json.loads((bundle / "quality.json").read_text())
        self.assertTrue(all(sample["passed"] for sample in quality["samples"]))
        semantic_result = json.loads(
            (bundle / "semantic-equivalence.json").read_text()
        )
        self.assertTrue(semantic_result["equivalent"])

    def test_native_worker_per_request_recreates_and_frees_engine(self):
        root = Path(tempfile.mkdtemp(prefix="tc-native-worker-lifecycle-"))
        config = root / "worker.json"
        config.write_text(json.dumps({
            "backend": "native",
            "engine_lifecycle": "per_request",
            "library": str(root / "fixture.dylib"),
            "model_id": "fixture",
            "model_path": str(root),
        }))
        parent, child = socket.socketpair()
        child_fd = child.detach()
        created: list[object] = []
        freed: list[object] = []

        class FakeLibrary:
            @staticmethod
            def tc_engine_free(engine):
                freed.append(engine)

        fake_library = FakeLibrary()

        def create_engine(_library, _config):
            engine = object()
            created.append(engine)
            return engine

        def run_sample(_library, _engine, command, lifecycle):
            return {
                "status": "success",
                "request_wall_seconds": 0.001,
                "denoise_seconds": 0.0005,
                "engine_lifecycle": lifecycle,
                "run_id_seen": command["run_id"],
            }

        with (
            mock.patch.object(
                campaign_runner, "load_native_library",
                return_value=fake_library,
            ),
            mock.patch.object(
                campaign_runner, "create_native_engine",
                side_effect=create_engine,
            ),
            mock.patch.object(
                campaign_runner, "run_native_sample",
                side_effect=run_sample,
            ),
        ):
            thread = threading.Thread(
                target=worker_main, args=(config, child_fd)
            )
            thread.start()
            try:
                ready = receive_message(parent, 2)
                self.assertEqual(ready["type"], "ready")
                self.assertEqual(
                    ready["engine_lifecycle"], "per_request"
                )
                responses = []
                for index in range(2):
                    send_message(parent, {
                        "type": "run",
                        "run_id": f"run-{index}",
                        "request": {},
                        "artifact_paths": {},
                    })
                    responses.append(receive_message(parent, 2))
                send_message(parent, {"type": "stop"})
                stopped = receive_message(parent, 2)
                self.assertEqual(stopped["type"], "stopped")
            finally:
                parent.close()
                thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(created), 2)
        self.assertEqual(freed, created)
        self.assertEqual(
            [response["engine_generation"] for response in responses],
            [1, 2],
        )
        self.assertEqual(
            len({response["worker_pid"] for response in responses}), 1
        )

    def test_schema_v1_v2_requests_share_one_semantic_identity(self):
        v1 = {
            "schema_version": 1,
            "model": "ltx-2.5-distilled",
            "operation": "video.generate",
            "prompt": "fox",
            "width": 64,
            "height": 64,
            "frames": 9,
            "fps": 24,
            "audio": False,
            "seed": 42,
            "steps": 11,
            "execution": "gpu",
            "ltx_backend": "c_metal",
            "ltx_fast_av": True,
            "residency": "streamed",
            "memory_budget_bytes": 123,
            "output": "/tmp/legacy.mp4",
        }
        v2 = {
            "schema_version": 2,
            "model": "ltx-2.5-distilled",
            "operation": "video.generate",
            "inputs": [
                {"kind": "text", "role": "prompt", "text": "fox"},
            ],
            "outputs": [{
                "kind": "video",
                "path": "/tmp/exact.mp4",
                "width": 64,
                "height": 64,
                "frames": 9,
                "fps": 24,
                "audio": False,
            }],
            "sampling": {"seed": 42, "steps": 11},
            "execution": {
                "policy": "gpu",
                "ltx_backend": "c_metal",
                "ltx_fast_av": True,
                "streaming": {"enabled": True},
            },
        }
        self.assertEqual(
            request_semantic_identity(v1),
            request_semantic_identity(v2),
        )

    def test_schema_v1_v2_image_requests_default_to_one_frame(self):
        v1 = {
            "schema_version": 1,
            "model": "z-image-turbo",
            "operation": "image.generate",
            "prompt": "fox",
            "width": 64,
            "height": 64,
            "seed": 42,
            "steps": 1,
            "execution": "gpu",
            "output": "/tmp/legacy.png",
        }
        v2 = {
            "schema_version": 2,
            "model": "z-image-turbo",
            "operation": "image.generate",
            "inputs": [
                {"kind": "text", "role": "prompt", "text": "fox"},
            ],
            "outputs": [{
                "kind": "image",
                "path": "/tmp/exact.png",
                "width": 64,
                "height": 64,
            }],
            "sampling": {"seed": 42, "steps": 1},
            "execution": {"policy": "gpu", "streaming": {"enabled": True}},
        }
        self.assertEqual(
            request_semantic_identity(v1),
            request_semantic_identity(v2),
        )

    def test_ltx_fill_counters_and_request_retention_normalize(self):
        common = {
            "stage": "denoiser",
            "resident_prefix_blocks": 8,
            "block_group_size": 1,
            "slot_count": 3,
            "prefetch_distance": 2,
            "io_workers": 3,
            "group_count": 40,
            "pass_count": 11,
            "startup_policy": "prefetch_window_before_prefix",
            "pass_transition": "reload",
            "reader_revision": 1,
            "weight_format": "convrot-int8-g256",
            "kernel_revision": "ltx-kernel-v1",
            "conditioning_recipe": "scalar-conditioning-v1",
            "upsample_boundary": "after-stage1-pool-retained",
        }
        legacy = {
            "block_streaming": {
                "actual_layout": {**common, "retention": "engine"},
                "request_slot_allocations": 3,
                "request_slot_refills": 437,
            }
        }
        exact = {
            "plan": {
                "streaming": {
                    "actual_layout": {**common, "retention": "request"},
                }
            },
            "block_streaming": {
                "request_slot_allocations": 3,
                "request_slot_refills": 440,
                "request_slot_fills": 440,
            },
        }
        legacy_semantic, legacy_missing = actual_semantic_layout(
            legacy, "per_request"
        )
        exact_semantic, exact_missing = actual_semantic_layout(
            exact, "per_request"
        )
        self.assertEqual(legacy_missing, [])
        self.assertEqual(exact_missing, [])
        self.assertEqual(legacy_semantic, exact_semantic)
        self.assertEqual(legacy_semantic["total_fills"], 440)
        persistent_legacy, _ = actual_semantic_layout(
            legacy, "persistent"
        )
        self.assertNotEqual(persistent_legacy, exact_semantic)

    def test_per_request_preserves_explicit_multi_pool_retention(self):
        result = {
            "plan": {
                "streaming": {
                    "actual_layout": {
                        "stage": "denoiser",
                        "resident_prefix_blocks": 0,
                        "block_group_size": 1,
                        "slot_count": 2,
                        "prefetch_distance": 1,
                        "io_workers": 2,
                        "group_count": 32,
                        "pass_count": 2,
                        "startup_policy": "prefetch_window_before_prefix",
                        "pass_transition": "reload",
                        "retention": "request;multi_pool=retain_all",
                        "reader_revision": 1,
                        "weight_format": "diffusers-bf16-sharded",
                        "kernel_revision": "flux-kernel-v1",
                        "conditioning_recipe": "flux-conditioning-v1",
                        "upsample_boundary": "no-upsample",
                    }
                }
            },
            "block_streaming": {"request_slot_fills": 64},
        }
        semantic, missing = actual_semantic_layout(result, "per_request")
        self.assertEqual(missing, [])
        self.assertEqual(semantic["retention"], "request;multi_pool=retain_all")

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

    def run_bundle(
        self, campaign: dict, audit: dict | None = None,
        environment: dict | None = None,
    ) -> Path:
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-test-"))
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(campaign))
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit or passed_audit()))
        environment_path = root / "environment.json"
        environment_path.write_text(json.dumps(environment or {
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

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin libproc")
    def test_p3_without_observed_swap_is_inconclusive(self):
        campaign = policy(blocks=10)
        campaign["comparison_kind"] = "P3"
        campaign["protocol"]["restart_workers_between_blocks"] = True
        campaign["memory_sampling"] = {
            "enabled": True,
            "interval_ms": 5,
            "max_gap_ms": 100,
            "required_variants": ["baseline", "candidate"],
            "allow_swap_out": True,
        }
        campaign["swap_comparison"] = {
            "revision": "tc-p3-natural-swap-v1",
            "pressure_source": "natural_low_memory_device",
            "baseline_role": "resident_or_default",
            "candidate_role": "public_streaming_exact",
            "minimum_baseline_swap_runs": 1,
            "candidate_swap_out_total_ratio_max": 1.0,
            "reporting_mode": "tradeoff_or_speedup",
        }
        for variant in ("baseline", "candidate"):
            campaign["variants"][variant]["source_identity"] = {
                "commit": "a" * 40,
                "source_manifest_sha256": "b" * 64,
                "clean": True,
            }
        environment = {
            "format": "turbocider-streaming-environment-v1",
            "status": "complete",
            "gpu": "synthetic",
            "ram_bytes": 64 << 30,
            "ssd": "synthetic",
            "power": "fixed",
            "thermal": "fixed",
            "pressure": {
                "protocol_revision": "tc-p3-natural-swap-v1",
                "source": "natural_low_memory_device",
                "state": "stable",
                "launched_by_runner": False,
                "cleanup_verified": True,
                "counter_scope": "host_global",
            },
        }
        bundle = self.run_bundle(campaign, environment=environment)
        result = verify(bundle)
        self.assertEqual(result["overall"], "INCONCLUSIVE")
        self.assertEqual(
            result["memory_evidence"]["p3_swap_status"], "INCONCLUSIVE"
        )
        self.assertEqual(result["p3_result"]["classification"], "not_qualified")

    def test_synthetic_abba_campaign_passes_with_persistent_worker_processes(self):
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
            generations = [
                row["engine_generation"] for row in rows
                if row["variant"] == variant
            ]
            self.assertEqual(len(set(generations)), len(generations))
        self.assertEqual(
            [row["variant"] for row in rows[:4]],
            ["baseline", "candidate", "candidate", "baseline"],
        )

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin libproc")
    def test_process_tree_memory_sampling_is_attached_to_each_request(self):
        campaign = policy(blocks=1)
        campaign["memory_sampling"] = {
            "enabled": True,
            "include_warmups": True,
            "interval_ms": 5,
            "max_gap_ms": 100,
            "root_role": "streaming-worker",
            "required_variants": ["baseline", "candidate"],
        }
        bundle = self.run_bundle(campaign)
        result = verify(bundle)
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(result["memory_evidence"]["qualification"], "PASS")
        rows = [
            json.loads(line)
            for line in (bundle / "memory-summaries.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["complete"] for row in rows))
        self.assertTrue(all(row["evidence_path"].startswith("memory/") for row in rows))

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin libproc")
    def test_p2_memory_evidence_applies_public_headroom_and_sample_floor(self):
        campaign = policy(blocks=10)
        campaign["comparison_kind"] = "P2"
        campaign["protocol"]["restart_workers_between_blocks"] = True
        campaign["memory_sampling"] = {
            "enabled": True,
            "interval_ms": 5,
            "max_gap_ms": 100,
            "required_variants": ["candidate"],
            "target_bytes": 8 << 30,
            "headroom_policy_revision": "tc-public-headroom-v1",
        }
        for variant in ("baseline", "candidate"):
            campaign["variants"][variant]["source_identity"] = {
                "commit": "a" * 40,
                "source_manifest_sha256": "b" * 64,
                "clean": True,
            }
        bundle = self.run_bundle(campaign)
        result = verify(bundle)
        memory = result["memory_evidence"]
        self.assertEqual(result["overall"], "PASS")
        self.assertEqual(memory["qualification"], "PASS")
        self.assertEqual(memory["required_count"], 20)
        target = 8 << 30
        self.assertEqual(
            memory["allowed_peak_bytes"],
            target - max(512 << 20, (target * 10 + 99) // 100),
        )
        self.assertEqual(memory["insufficient_variants"], [])
        self.assertEqual(memory["fresh_process_generations"]["candidate"], 10)
        self.assertEqual(memory["fresh_process_failures"], [])
        self.assertLessEqual(
            memory["peak_p95_bytes"]["candidate"],
            memory["allowed_peak_bytes"],
        )

    def test_p1_rejects_semantically_different_layouts_with_same_digest(self):
        campaign = policy(blocks=1)
        campaign["variants"]["candidate"]["synthetic"][
            "actual_semantic_layout"
        ]["io_workers"] = 1
        bundle = self.run_bundle(campaign)
        semantic = json.loads(
            (bundle / "semantic-equivalence.json").read_text()
        )
        self.assertFalse(semantic["equivalent"])
        with self.assertRaises(EvidenceError):
            verify(bundle)

    def test_p1_rejects_equal_layouts_that_differ_from_frozen_expected(self):
        campaign = policy(blocks=1)
        for variant in ("baseline", "candidate"):
            campaign["variants"][variant]["synthetic"][
                "actual_semantic_layout"
            ]["resident_prefix_blocks"] = 9
        bundle = self.run_bundle(campaign)
        semantic = json.loads(
            (bundle / "semantic-equivalence.json").read_text()
        )
        self.assertTrue(semantic["observed_same_semantic_layout"])
        self.assertFalse(semantic["observed_matches_expected"])
        with self.assertRaises(EvidenceError):
            verify(bundle)

    def test_p1_rejects_incomplete_actual_semantics(self):
        campaign = policy(blocks=1)
        del campaign["variants"]["candidate"]["synthetic"][
            "actual_semantic_layout"
        ]["startup_policy"]
        bundle = self.run_bundle(campaign)
        with self.assertRaises(EvidenceError):
            verify(bundle)

    def test_expected_implementations_are_checked_per_request(self):
        campaign = policy(blocks=1)
        campaign["expected_implementations"] = {
            "baseline": "direct-v1",
            "candidate": "generic-v1",
        }
        campaign["variants"]["baseline"]["synthetic"][
            "streaming_implementation"
        ] = "direct-v1"
        campaign["variants"]["candidate"]["synthetic"][
            "streaming_implementation"
        ] = "generic-v1"
        bundle = self.run_bundle(campaign)
        result = verify(bundle)
        self.assertEqual(result["overall"], "PASS")
        rows = [
            json.loads(line)
            for line in (bundle / "raw-samples.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            {row["streaming_implementation"] for row in rows},
            {"direct-v1", "generic-v1"},
        )

        rows[0]["streaming_implementation"] = "generic-v1"
        raw_path = bundle / "raw-samples.jsonl"
        raw_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["files"]["raw-samples.jsonl"]["sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        manifest["files"]["raw-samples.jsonl"]["bytes"] = (
            raw_path.stat().st_size
        )
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(
            EvidenceError, "streaming implementation differs"
        ):
            verify(bundle)

    def test_policy_rejects_incomplete_expected_implementations(self):
        campaign = policy(blocks=1)
        campaign["expected_implementations"] = {
            "candidate": "generic-v1",
        }
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-implementation-"))
        path = root / "policy.json"
        path.write_text(json.dumps(campaign))
        with self.assertRaisesRegex(CampaignError, "expected_implementations"):
            run_campaign(path, root / "bundle")

    def test_p1_rejects_threshold_looser_than_two_percent(self):
        campaign = policy(blocks=1)
        campaign["thresholds"]["P1_same_layout"][
            "wall_median_ratio_max"
        ] = 1.03
        bundle = self.run_bundle(campaign)
        with self.assertRaises(EvidenceError):
            verify(bundle)

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

    def test_p2_requires_public_memory_sampling_contract(self):
        campaign = policy(blocks=1)
        campaign["comparison_kind"] = "P2"
        campaign["protocol"]["restart_workers_between_blocks"] = True
        with self.assertRaises(CampaignError):
            campaign_runner.validate_policy(campaign)
        campaign["memory_sampling"] = {
            "enabled": True,
            "target_bytes": 7 << 30,
            "headroom_policy_revision": "tc-public-headroom-v1",
        }
        with self.assertRaises(CampaignError):
            campaign_runner.validate_policy(campaign)
        campaign["memory_sampling"]["target_bytes"] = 8 << 30
        campaign_runner.validate_policy(campaign)

    def test_p3_requires_natural_swap_contract_and_both_variants(self):
        campaign = policy(blocks=10)
        campaign["comparison_kind"] = "P3"
        campaign["protocol"]["restart_workers_between_blocks"] = True
        campaign["memory_sampling"] = {
            "enabled": True,
            "interval_ms": 20,
            "max_gap_ms": 100,
            "required_variants": ["baseline", "candidate"],
            "allow_swap_out": True,
        }
        campaign["swap_comparison"] = {
            "revision": "tc-p3-natural-swap-v1",
            "pressure_source": "externally_managed_fixed_pressure",
            "baseline_role": "resident_or_default",
            "candidate_role": "public_streaming_exact",
            "minimum_baseline_swap_runs": 1,
            "candidate_swap_out_total_ratio_max": 1.0,
            "reporting_mode": "tradeoff_or_speedup",
        }
        for variant in ("baseline", "candidate"):
            campaign["variants"][variant]["source_identity"] = {
                "commit": "a" * 40,
                "source_manifest_sha256": "b" * 64,
                "clean": True,
            }
        campaign_runner.validate_policy(campaign)

        invalid = json.loads(json.dumps(campaign))
        invalid["memory_sampling"]["required_variants"] = ["candidate"]
        with self.assertRaises(CampaignError):
            campaign_runner.validate_policy(invalid)

        invalid = json.loads(json.dumps(campaign))
        invalid["swap_comparison"]["pressure_source"] = "developer_laptop_pressure"
        with self.assertRaises(CampaignError):
            campaign_runner.validate_policy(invalid)

    def test_worker_launch_order_is_frozen_and_recorded(self):
        campaign = policy(blocks=1)
        campaign["protocol"]["worker_launch_order"] = [
            "candidate", "baseline",
        ]
        bundle = self.run_bundle(campaign)
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(
            manifest["worker_launch_order"], ["candidate", "baseline"]
        )

        campaign["protocol"]["worker_launch_order"] = [
            "candidate", "candidate",
        ]
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-policy-"))
        path = root / "policy.json"
        path.write_text(json.dumps(campaign))
        with self.assertRaisesRegex(CampaignError, "worker_launch_order"):
            run_campaign(path, root / "bundle")

    def test_policy_rejects_unknown_engine_lifecycle(self):
        campaign = policy(blocks=1)
        campaign["engine_lifecycle"] = "sometimes"
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-lifecycle-"))
        path = root / "policy.json"
        path.write_text(json.dumps(campaign))
        with self.assertRaises(CampaignError):
            run_campaign(path, root / "bundle")

    def test_p1_policy_requires_frozen_expected_actual(self):
        campaign = policy(blocks=1)
        del campaign["semantic_equivalence"]["expected_actual"]
        root = Path(tempfile.mkdtemp(prefix="tc-campaign-expected-"))
        path = root / "policy.json"
        path.write_text(json.dumps(campaign))
        with self.assertRaises(CampaignError):
            run_campaign(path, root / "bundle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
