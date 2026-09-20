#!/usr/bin/env python3
"""Contract tests for the test/calibration-only public streaming catalog."""

from __future__ import annotations

import copy
import ctypes as c
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

import build_streaming_catalog as builder  # noqa: E402
from test_z_image_streaming_descriptor import write_fixture  # noqa: E402


GIB = 1 << 30
TEST_SYMBOLS = (
    "tc_engine_test_set_streaming_catalog_json",
    "tc_engine_test_clear_streaming_catalog",
    "tc_engine_test_build_streaming_catalog_json",
)


def consume(library, pointer: c.c_void_p) -> str:
    if not pointer.value:
        return ""
    value = c.string_at(pointer).decode()
    library.tc_string_free(pointer)
    return value


def byte_vocab() -> dict[str, int]:
    result: dict[str, int] = {}
    extra = 0
    for value in range(256):
        if 33 <= value <= 126 or 161 <= value <= 172 or value >= 174:
            codepoint = value
        else:
            codepoint = 256 + extra
            extra += 1
        result[chr(codepoint)] = value
    return result


def record() -> dict:
    value = {
        "id": "tc-test-zimage-mismatched-workload",
        "revision": 1,
        "catalog_revision": "tc-streaming-test-catalog-r1",
        "source": {
            "model_variant": "z-image-turbo-bf16",
            "weight_format": "safetensors-bf16",
            "artifact_manifest_digest": "a" * 64,
            "source_snapshot_digest": "b" * 64,
        },
        "workload": {
            "model": "z-image-turbo",
            "operation": "image.generate",
            "execution": "gpu",
            # Deliberately differs from the real probe. Resolution must move
            # beyond the empty-catalog gate and reject this workload.
            "device_class": "test-device-class",
            "execution_container": "embedded_app",
            "width": 64,
            "height": 64,
            "frames": 1,
            "fps": 0,
            "steps": 1,
            "batch": 1,
            "audio": False,
            "dynamic_text": True,
            "approximation": False,
            "conditioning_revision": "qwen3-simple-flow-shift3-v1",
            "vae_policy_revision": "z-image-vae-v1",
            "feature_digest": "c" * 64,
            "token_shapes": [{
                "encoder": "qwen3",
                "tokenizer_revision": "qwen3-z-image-v1",
                "template_revision": "z-image-template-v1",
                "valid_rows": 4,
                "padded_rows": 128,
                "compute_rows": 128,
            }],
        },
        "runtime": {
            "turbocider_build_id": "test-build",
            "runtime_revision": "public-streaming-runtime-v2",
            "adapter_revision": "z-image-public-adapter-v2-k1-k2",
            "reader_revision": "z-image-reader-test-v1",
            "kernel_revision": "z-image-kernel-test-v1",
            "allocator_policy_revision": (
                "mlx-request-cache-policy-v2-k1-zero-cache"
            ),
        },
        "device": {
            "minimum_physical_memory_bytes": 1,
            "maximum_physical_memory_bytes": 0,
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
                        "resident_prefix_blocks": 0,
                        "prefetch_distance": 0,
                        "io_workers": 1,
                    }
                },
            },
            "layout_digest": "d" * 64,
            "component_policy_revision": "z-image-components-test-v1",
            "pass_transition": "reload",
            "multi_pool_policy": "serial",
        },
        "calibration": {
            "complete": True,
            "calibrated_request_bytes": 8 * GIB,
            "scope": "execution_process_tree_v1",
            "estimator_revision": "tree-phys-footprint-linear-p95-v1",
            "calibration_id": "test-calibration-v1",
            "execution_container": "embedded_app",
            "evidence_digest": "e" * 64,
            "confirmation_sample_count": 20,
            "maximum_sample_gap_ns": 20_000_000,
        },
        "performance": {
            "rank": 1,
            "logical_read_bytes": 1,
            "profile_id": "test-performance-v1",
            "comparison_kind": "P1_same_layout",
            "confidence_status": "PASS",
            "evidence_digest": "f" * 64,
        },
        "release": {
            "channel": "public-experimental",
            "revoked": False,
            "reviewed_commit": "test-review-commit",
            "review_digest": "1" * 64,
        },
    }
    value["canonical_record_digest"] = builder.canonical_record_digest(value)
    return value


def catalog() -> dict:
    return {
        "schema": "turbocider-streaming-test-catalog-v1",
        "revision": "tc-streaming-test-catalog-r1",
        "records": [record()],
    }


def request() -> dict:
    return {
        "schema_version": 2,
        "model": "z-image-turbo",
        "operation": "image.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": "gate"}],
        "outputs": [{
            "kind": "image",
            "path": "/tmp/not-generated-test-catalog.png",
            "width": 64,
            "height": 64,
        }],
        "sampling": {"seed": 42, "steps": 1},
        "execution": {
            "policy": "gpu",
            "streaming": {
                "schema_version": 2,
                "enabled": True,
                "selection": "memory_tier",
                "retention": "request",
                "target_request_memory_bytes": 12 * GIB,
            },
        },
    }


class TestCatalogTests(unittest.TestCase):
    def test_release_surface_does_not_declare_test_catalog_hooks(self):
        header = (ROOT / "bindings/c/include/turbocider/turbocider.h").read_text()
        api = (ROOT / "native/api/c_api.mm").read_text()
        build = (ROOT / "tools/native/build.sh").read_text()
        for symbol in TEST_SYMBOLS:
            self.assertNotIn(symbol, header)
            self.assertIn(symbol, api)
        self.assertIn("TURBOCIDER_ENABLE_TEST_HOOKS", api)
        self.assertIn("TURBOCIDER_BUILD_TEST_HOOKS", build)
        self.assertIn("streaming_catalog_test.mm", build)

    def test_release_library_does_not_export_test_catalog_hooks(self):
        raw = os.environ.get("TURBOCIDER_RELEASE_LIBRARY")
        if not raw:
            self.skipTest("TURBOCIDER_RELEASE_LIBRARY was not provided")
        symbols = subprocess.check_output(
            ["nm", "-gU", raw], text=True, stderr=subprocess.STDOUT
        )
        for symbol in TEST_SYMBOLS:
            self.assertNotIn(symbol, symbols)

    def test_hook_build_installs_engine_scoped_immutable_catalog(self):
        raw = os.environ.get("TURBOCIDER_TEST_HOOK_LIBRARY")
        if not raw:
            self.skipTest("TURBOCIDER_TEST_HOOK_LIBRARY was not provided")
        library = c.CDLL(raw)
        library.tc_string_free.argtypes = [c.c_void_p]
        library.tc_engine_create_model.argtypes = [
            c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p),
        ]
        library.tc_engine_create_model_candidate.argtypes = (
            library.tc_engine_create_model.argtypes
        )
        library.tc_engine_create_model_worker.argtypes = library.tc_engine_create_model.argtypes

        library.tc_engine_resolve_streaming_json.argtypes = [
            c.c_void_p, c.c_char_p, c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p),
        ]
        library.tc_engine_test_set_streaming_catalog_json.argtypes = [
            c.c_void_p, c.c_char_p, c.POINTER(c.c_void_p),
        ]
        library.tc_engine_test_clear_streaming_catalog.argtypes = [
            c.c_void_p, c.POINTER(c.c_void_p),
        ]
        library.tc_engine_test_build_streaming_catalog_json.argtypes = [
            c.c_void_p, c.c_char_p, c.c_char_p, c.c_uint64, c.c_char_p,
            c.POINTER(c.c_void_p), c.POINTER(c.c_void_p),
        ]
        library.tc_engine_free.argtypes = [c.c_void_p]
        library.tc_engine_verify_streaming_sources_json.argtypes = [
            c.c_void_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]

        with tempfile.TemporaryDirectory(prefix="tc-test-catalog-") as raw_root:
            root = Path(raw_root)
            for relative in (
                "split_files/text_encoders/qwen_3_4b.safetensors",
                "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                "split_files/vae/ae.safetensors",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if "diffusion_models" in relative:
                    write_fixture(path)
                else:
                    path.write_bytes(b"test-catalog-source")
            tokenizer = root / "tokenizer/tokenizer.json"
            tokenizer.parent.mkdir(parents=True)
            tokenizer.write_text(json.dumps({
                "model": {"type": "BPE", "vocab": byte_vocab(), "merges": []},
                "pre_tokenizer": {
                    "pretokenizers": [{"pattern": {"Regex": "."}}]
                },
                "added_tokens": [],
            }))

            def create(constructor) -> c.c_void_p:
                engine, error = c.c_void_p(), c.c_void_p()
                status = constructor(
                    b"z-image-turbo", str(root).encode(),
                    c.byref(engine), c.byref(error),
                )
                failure = consume(library, error)
                self.assertEqual(status, 0, failure)
                self.assertTrue(engine.value)
                return engine

            def install(engine: c.c_void_p, value: dict) -> tuple[int, str]:
                error = c.c_void_p()
                status = library.tc_engine_test_set_streaming_catalog_json(
                    engine, json.dumps(value).encode(), c.byref(error)
                )
                return status, consume(library, error)

            def resolve_full(engine: c.c_void_p) -> tuple[int, str, str]:
                result, error = c.c_void_p(), c.c_void_p()
                status = library.tc_engine_resolve_streaming_json(
                    engine, json.dumps(request()).encode(),
                    c.byref(result), c.byref(error),
                )
                return (
                    status, consume(library, result), consume(library, error)
                )

            def resolve(engine: c.c_void_p) -> tuple[int, str]:
                status, _result, failure = resolve_full(engine)
                return status, failure

            def build_exact_catalog(engine: c.c_void_p) -> dict:
                plan = {
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
                    "pass_transition": "reload",
                    "multi_pool_policy": "serial",
                }
                output, error = c.c_void_p(), c.c_void_p()
                status = library.tc_engine_test_build_streaming_catalog_json(
                    engine, json.dumps(request()).encode(),
                    json.dumps(plan).encode(), 12 * GIB,
                    b"tc-streaming-test-generated-r1",
                    c.byref(output), c.byref(error),
                )
                value, failure = consume(library, output), consume(library, error)
                self.assertEqual(status, 0, failure)
                self.assertTrue(value)
                return json.loads(value)

            def build_exact_catalog_with_cli() -> dict:
                request_path = root / "request.json"
                plan_path = root / "plan.json"
                output_path = root / "generated-test-catalog.json"
                request_path.write_text(json.dumps(request()))
                plan_path.write_text(json.dumps({
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
                    "pass_transition": "reload",
                    "multi_pool_policy": "serial",
                }))
                subprocess.run([
                    sys.executable, "-B",
                    str(ROOT / "tools/native/build_test_streaming_catalog.py"),
                    "--library", raw,
                    "--model-id", "z-image-turbo",
                    "--model-path", str(root),
                    "--request", str(request_path),
                    "--plan", str(plan_path),
                    "--target-gib", "12",
                    "--catalog-revision", "tc-streaming-test-cli-r1",
                    "--output", str(output_path),
                ], check=True, capture_output=True, text=True)
                return json.loads(output_path.read_text())

            public = create(library.tc_engine_create_model)
            try:
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("catalog_has_no_public_records", failure)

                generated = build_exact_catalog(public)
                self.assertEqual(
                    generated["schema"],
                    "turbocider-streaming-test-catalog-v1",
                )
                generated_record = generated["records"][0]
                self.assertEqual(
                    generated_record["plan"]["canonical_config"]
                    ["stages"]["denoiser"]["resident_prefix_blocks"],
                    14,
                )
                self.assertEqual(
                    len(generated_record["plan"]["layout_digest"]), 64
                )
                status, failure = install(public, generated)
                self.assertEqual(status, 0, failure)
                status, resolved, failure = resolve_full(public)
                self.assertEqual(status, 0, failure)
                resolved_value = json.loads(resolved)
                self.assertEqual(resolved_value["status"], "resolved")
                self.assertEqual(
                    resolved_value["selection"]["layout_digest"],
                    generated_record["plan"]["layout_digest"],
                )
                cli_generated = build_exact_catalog_with_cli()
                self.assertEqual(cli_generated["records"][0]["workload"]["execution_container"], "cli_worker")
                self.assertEqual(
                    cli_generated["records"][0]["plan"]["layout_digest"],
                    generated_record["plan"]["layout_digest"],
                )

                portable = copy.deepcopy(generated)
                portable_record = portable["records"][0]
                portable_record["source"]["identity_version"] = 2
                del portable_record["source"]["source_snapshot_digest"]
                portable_record["canonical_record_digest"] = builder.canonical_record_digest(portable_record)
                status, failure = install(public, portable)
                self.assertEqual(status, 0, failure)
                # Portable records cannot authorize the still-legacy adapter.
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("artifact_verification_required", failure)
                invalid_portable = copy.deepcopy(portable)
                invalid_portable["records"][0]["source"]["source_snapshot_digest"] = "b" * 64
                status, failure = install(public, invalid_portable)
                self.assertNotEqual(status, 0)
                self.assertIn("missing or unknown fields", failure)

                error = c.c_void_p()
                status = library.tc_engine_test_clear_streaming_catalog(
                    public, c.byref(error)
                )
                failure = consume(library, error)
                self.assertEqual(status, 0, failure)

                status, failure = install(public, catalog())
                self.assertEqual(status, 0, failure)
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("unvalidated_workload", failure)
                self.assertNotIn("catalog_has_no_public_records", failure)

                malformed = copy.deepcopy(catalog())
                malformed["unknown"] = True
                status, failure = install(public, malformed)
                self.assertNotEqual(status, 0)
                self.assertIn("missing or unknown fields", failure)
                # Failed replacement must leave the previously installed
                # immutable snapshot active.
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("unvalidated_workload", failure)

                wrong_digest = copy.deepcopy(catalog())
                wrong_digest["records"][0]["canonical_record_digest"] = "0" * 64
                status, failure = install(public, wrong_digest)
                self.assertNotEqual(status, 0)
                self.assertIn("canonical record digest mismatch", failure)

                error = c.c_void_p()
                status = library.tc_engine_test_clear_streaming_catalog(
                    public, c.byref(error)
                )
                failure = consume(library, error)
                self.assertEqual(status, 0, failure)
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("catalog_has_no_public_records", failure)
                def verify():
                    result, error = c.c_void_p(), c.c_void_p()
                    status = library.tc_engine_verify_streaming_sources_json(
                        public, c.byref(result), c.byref(error))
                    failure = consume(library, error)
                    self.assertEqual(status, 0, failure)
                    return json.loads(consume(library, result))

                proof = verify()
                self.assertEqual(proof["status"], "verified")
                self.assertEqual(len(proof["files"]), 4)
                self.assertGreater(proof["verification_bytes_read"], 0)
                cached = verify()
                self.assertEqual(cached["verification_bytes_read"], 0)
                self.assertEqual(cached["verification_cache_hits"], 4)
                verified_catalog = build_exact_catalog(public)
                verified_record = verified_catalog["records"][0]
                self.assertEqual(verified_record["source"]["identity_version"], 2)
                self.assertNotIn("source_snapshot_digest", verified_record["source"])
                self.assertEqual(verified_record["source"]["artifact_manifest_digest"],
                                 proof["artifact_manifest_digest"])
                self.assertEqual(verified_record["canonical_record_digest"],
                                 builder.canonical_record_digest(verified_record))
                status, failure = install(public, verified_catalog)
                self.assertEqual(status, 0, failure)
                status, failure = resolve(public)
                self.assertEqual(status, 0, failure)
                text_file = root / "split_files/text_encoders/qwen_3_4b.safetensors"
                with text_file.open("r+b") as stream:
                    stream.write(b"X")
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("artifact_verification_required", failure)
                updated = verify()
                self.assertNotEqual(updated["artifact_manifest_digest"], proof["artifact_manifest_digest"])
                status, failure = resolve(public)
                self.assertNotEqual(status, 0)
                self.assertIn("artifact_verification_required", failure)
            finally:
                library.tc_engine_free(public)

            worker = create(library.tc_engine_create_model_worker)
            try:
                worker_catalog = build_exact_catalog(worker)
                worker_record = worker_catalog["records"][0]
                self.assertEqual(worker_record["workload"]["execution_container"], "cli_worker")
                self.assertEqual(worker_record["calibration"]["execution_container"], "cli_worker")
                status, failure = install(worker, worker_catalog)
                self.assertEqual(status, 0, failure)
                status, failure = resolve(worker)
                self.assertEqual(status, 0, failure)
                app_catalog = copy.deepcopy(worker_catalog)
                app_record = app_catalog["records"][0]
                app_record["workload"]["execution_container"] = "embedded_app"
                app_record["calibration"]["execution_container"] = "embedded_app"
                app_record["canonical_record_digest"] = builder.canonical_record_digest(app_record)
                status, failure = install(worker, app_catalog)
                self.assertEqual(status, 0, failure)
                status, failure = resolve(worker)
                self.assertNotEqual(status, 0)
                self.assertIn("unvalidated_workload", failure)
            finally:
                library.tc_engine_free(worker)

            candidate = create(library.tc_engine_create_model_candidate)
            try:
                status, failure = install(candidate, catalog())
                self.assertNotEqual(status, 0)
                self.assertIn("requires a public engine", failure)
            finally:
                library.tc_engine_free(candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
