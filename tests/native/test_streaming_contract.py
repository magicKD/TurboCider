#!/usr/bin/env python3
"""Request-level layout-first tests against the freshly built native library."""
import copy
import ctypes as C
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = C.CDLL(str(ROOT / "build/native/libturbocider.dylib"))
LIB.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
LIB.tc_streaming_options_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
LIB.tc_string_free.argtypes = [C.c_void_p]


def plan(value):
    out, err = C.c_void_p(), C.c_void_p()
    raw = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    status = LIB.tc_plan_json(raw, C.byref(out), C.byref(err))
    result = json.loads(C.string_at(out).decode()) if out.value else None
    error = C.string_at(err).decode() if err.value else ""
    LIB.tc_string_free(out)
    LIB.tc_string_free(err)
    return status, result, error


def options(value):
    out, err = C.c_void_p(), C.c_void_p()
    raw = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    status = LIB.tc_streaming_options_json(raw, C.byref(out), C.byref(err))
    result = json.loads(C.string_at(out).decode()) if out.value else None
    error = C.string_at(err).decode() if err.value else ""
    LIB.tc_string_free(out)
    LIB.tc_string_free(err)
    return status, result, error


def request():
    return {
        "schema_version": 2, "model": "ltx-2.5-distilled", "operation": "video.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": "test"}],
        "outputs": [{"kind": "video", "path": "/tmp/streaming-contract-not-generated.mp4",
                     "width": 64, "height": 64, "frames": 9, "audio": False}],
        "sampling": {"steps": 11},
        "execution": copy.deepcopy(json.loads((ROOT / "docs/design/streaming/examples/manual-ltx.json").read_text())["execution"]),
    }


def selector_request(target=12 << 30):
    value = request()
    value["execution"]["streaming"] = {
        "schema_version": 2,
        "enabled": True,
        "selection": "memory_tier",
        "retention": "request",
        "target_request_memory_bytes": target,
    }
    return value


class StreamingContract(unittest.TestCase):
    def test_manual_hybrid_is_model_owned(self):
        r = {
            "schema_version": 2, "model": "flux2-klein-4b", "operation": "image.generate",
            "inputs": [{"kind": "text", "role": "prompt", "text": "test"}],
            "outputs": [{"kind": "image", "path": "/tmp/not-generated.png", "width": 256, "height": 256}],
            "sampling": {"steps": 8},
            "execution": {"policy": "gpu_ane", "allow_approximation": True,
                "ane_manifest": "/tmp/not-opened-manual-hybrid.json",
                "streaming": {"enabled": True, "schema_version": 1, "selection": "manual", "retention": "request",
                    "stages": {"denoiser": {"residency": "streamed", "block_group_size": 1, "slot_count": 3,
                        "resident_prefix_blocks": 10, "prefetch_distance": 2, "io_workers": 2}}}}}
        status, _, error = plan(r)
        self.assertNotEqual(status, 0)
        self.assertIn("GPU-only", error)
        r["model"] = "z-image-turbo"
        status, _, error = plan(r)
        # Unqualified hardware must be rejected by the model capability.
        if status:
            self.assertIn("measured-device GPU+ANE", error)
        r["execution"]["encoder_ane_manifest"] = "/tmp/not-opened-encoder.json"
        status, _, error = plan(r)
        self.assertNotEqual(status, 0)
        self.assertIn("without encoder ANE", error)

    def test_public_selector_is_plan_only_until_engine_resolution(self):
        r = selector_request()
        status, p, error = plan(r)
        self.assertEqual(status, 0, error)
        report = p["streaming"]
        self.assertEqual(report["requested_selector"], r["execution"]["streaming"])
        self.assertEqual(report["merged_selector"], r["execution"]["streaming"])
        self.assertFalse(report["execution_supported"])
        self.assertEqual(report["rejection_code"], "streaming_preset_resolution_required")
        self.assertEqual(p["memory_estimate_kind"], "requires_preset_resolution")

    def test_public_selector_disabled_is_same_legacy_plan(self):
        r = request(); del r["execution"]["streaming"]
        a, baseline, error = plan(r); self.assertEqual(a, 0, error)
        r["execution"]["streaming"] = {"schema_version": 2, "enabled": False}
        b, disabled, error = plan(r); self.assertEqual(b, 0, error)
        self.assertEqual(disabled, baseline)

    def test_public_selector_validation_and_conflicts(self):
        for patch in [
            {"target_request_memory_bytes": 0},
            {"target_request_memory_bytes": 14 << 30},
            {"target_request_memory_bytes": True},
            {"target_request_memory_bytes": 1.5},
            {"selection": "unknown"},
            {"selection": "preset"},
            {"preset_id": "unexpected"},
            {"stages": {}},
        ]:
            r = selector_request(); r["execution"]["streaming"].update(patch)
            self.assertNotEqual(plan(r)[0], 0, patch)
        r = selector_request(); r["execution"]["memory_budget_bytes"] = 0
        self.assertIn("streaming_config_conflict", plan(r)[2])
        r = selector_request(); r["execution"]["memory_constrained"] = {
            "enabled": True, "limit_bytes": 16 << 30,
        }
        self.assertIn("streaming_config_conflict", plan(r)[2])
        raw = json.dumps(selector_request()).replace(
            str(12 << 30), "1.2884901888e10", 1)
        self.assertIn("unsigned decimal integer syntax", plan(raw)[2])

    def test_public_options_are_metadata_only_and_fail_closed(self):
        status, result, error = options(selector_request())
        self.assertEqual(status, 0, error)
        self.assertEqual(result["catalog_revision"], "tc-streaming-catalog-empty-v1")
        self.assertEqual(result["query_status"], "catalog_empty")
        self.assertEqual(len(result["targets"]), 5)
        self.assertTrue(all(item["status"] == "catalog_empty" for item in result["targets"]))
        self.assertTrue(all(item["reason_code"] == "catalog_has_no_public_records"
                            for item in result["targets"]))

    def test_manual_is_explicit_plan_only(self):
        r = request()
        status, p, error = plan(r)
        self.assertEqual(status, 0, error)
        report = p["streaming"]
        self.assertEqual(report["requested_layout"], r["execution"]["streaming"])
        self.assertFalse(report["execution_supported"])
        self.assertFalse(p["executable"])
        self.assertEqual(report["merged_intent"], r["execution"]["streaming"])
        self.assertIsNone(report["resolved_layout"])
        self.assertIsNone(report["actual_layout"])
        self.assertEqual(report["field_provenance"]["stages.denoiser.slot_count"], "request")

    def test_budget_does_not_reselect_slots(self):
        r = request()
        r["execution"]["memory_constrained"] = {
            "enabled": True, "limit_bytes": 8 << 30, "buffer_percent": 15,
            "min_free_bytes": 1 << 30,
        }
        status, p, error = plan(r)
        self.assertEqual(status, 0, error)
        self.assertEqual(p["streaming"]["requested_layout"], r["execution"]["streaming"])
        self.assertEqual(p["memory_policy"]["denoiser_budget_bytes"], 0)

    def test_disabled_is_same_legacy_plan(self):
        r = request(); del r["execution"]["streaming"]
        a, baseline, error = plan(r); self.assertEqual(a, 0, error)
        r["execution"]["streaming"] = {"enabled": False}
        b, disabled, error = plan(r); self.assertEqual(b, 0, error)
        self.assertEqual(disabled, baseline)

    def test_conflicts_even_zero_and_default_values(self):
        for key, value in [("residency", "component_staged"), ("memory_budget_bytes", 0)]:
            r = request(); r["execution"][key] = value
            status, _, error = plan(r)
            self.assertNotEqual(status, 0)
            self.assertIn("streaming_config_conflict", error)
        r = request(); r["parameters"] = {"streaming_offload": False}
        self.assertIn("streaming_config_conflict", plan(r)[2])

    def test_type_range_and_unknown_fields(self):
        for key, value in [("slot_count", True), ("slot_count", 0), ("slot_count", -1),
                           ("slot_count", 1.5), ("slot_count", 1 << 40),
                           ("prefetch_distance", 3), ("io_workers", 4), ("typo", 1)]:
            r = request(); r["execution"]["streaming"]["stages"]["denoiser"][key] = value
            self.assertNotEqual(plan(r)[0], 0, (key, value))

    def test_explicit_legacy_slot_cap(self):
        r = request()
        r["execution"]["memory_constrained"] = {"enabled": True, "limit_bytes": 16 << 30,
                                                 "max_refill_slots": 2}
        self.assertIn("max_refill_slots", plan(r)[2])

    def test_resident_has_no_streamed_fields(self):
        r = request(); s = r["execution"]["streaming"]["stages"]
        s["denoiser"] = {"residency": "resident"}
        self.assertEqual(plan(r)[0], 0)
        s["denoiser"]["slot_count"] = 0
        self.assertNotEqual(plan(r)[0], 0)

    def test_duplicate_decoded_keys(self):
        raw = json.dumps(request()).replace('"slot_count": 3', '"slot_count": 3, "slot_count": 2')
        self.assertIn("duplicate JSON", plan(raw)[2])
        raw = json.dumps(request()).replace('"enabled": true', '"enabled": true, "enabl\\u0065d": false')
        self.assertIn("duplicate JSON", plan(raw)[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
