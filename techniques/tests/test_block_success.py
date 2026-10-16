"""Offline raw-pair contracts; synthetic fixtures never become device evidence."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
import block_context
import block_success as success


def raw_header(key="input"):
    return {
        key: {"dtype": "BF16", "shape": [1, 1024, 4096],
              "data_offsets": [0, success.PAYLOAD_BYTES]},
        "__metadata__": {"recipe": success.RECIPE},
    }


def encoded_header(value):
    data = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return data + b" " * (-len(data) % 8)


class RawBF16Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Full declared geometry, but wholly synthetic bytes: no model fixture.
        cls.payload = struct.pack("<4H", 0, 0x3F80, 0xBF80, 0x7F7F) * (success.ELEMENTS // 4)

    def parse(self, header=None, payload=None, key="input", prefix=None):
        header = encoded_header(raw_header(key)) if header is None else header
        payload = self.payload if payload is None else payload
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.safetensors"
            path.write_bytes((struct.pack("<Q", len(header)) if prefix is None else prefix) + header + payload)
            return success.raw_bf16(path, key)

    def test_full_finite_input_and_reference_bind_file_and_payload_hashes(self):
        for key in ("input", "reference"):
            header = encoded_header(raw_header(key))
            metadata, payload = self.parse(header=header, key=key)
            self.assertEqual(payload, self.payload)
            self.assertEqual(metadata["shape"], [1, 1024, 4096])
            self.assertEqual(metadata["payload_bytes"], 8 * 1024**2)
            self.assertEqual(metadata["payload_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(metadata["file_sha256"], hashlib.sha256(struct.pack("<Q", len(header)) + header + payload).hexdigest())
            self.assertEqual(metadata["file_bytes"], 8 + len(header) + len(payload))
            self.assertTrue(metadata["all_finite"])

    def test_bad_dtype_shape_offsets_recipe_and_boolean_integers_are_rejected(self):
        for kind in ("dtype", "shape", "offset", "recipe", "boolean-shape", "boolean-offset", "extra"):
            header = raw_header()
            if kind == "dtype":
                header["input"]["dtype"] = "F16"
            elif kind == "shape":
                header["input"]["shape"] = [1024, 4096]
            elif kind == "offset":
                header["input"]["data_offsets"][1] -= 2
            elif kind == "recipe":
                header["__metadata__"]["recipe"] = "unbound"
            elif kind == "boolean-shape":
                header["input"]["shape"][0] = True
            elif kind == "boolean-offset":
                header["input"]["data_offsets"][0] = False
            else:
                header["reference"] = copy.deepcopy(header["input"])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.parse(header=encoded_header(header))

    def test_short_trailing_and_unbounded_or_unaligned_headers_are_rejected(self):
        header = encoded_header(raw_header())
        for prefix, body, payload in (
            (b"short", header, self.payload),
            (struct.pack("<Q", 65544), header, self.payload),
            (struct.pack("<Q", len(header) - 1), header, self.payload),
            (None, header, self.payload[:-1]),
            (None, header, self.payload + b"extra"),
        ):
            with self.subTest(prefix=prefix, payload_bytes=len(payload)), self.assertRaises(ValueError):
                self.parse(header=body, payload=payload, prefix=prefix)

    def test_duplicate_header_keys_and_implicit_tensor_selection_are_rejected(self):
        header = encoded_header(raw_header()).rstrip(b" ")
        duplicate = header[:-1] + b',"__metadata__":{"recipe":"' + success.RECIPE.encode() + b'"}}'
        duplicate += b" " * (-len(duplicate) % 8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.parse(header=duplicate)
        with self.assertRaises(ValueError):
            self.parse(key="automatic")
        with self.assertRaises(ValueError):
            self.parse(header=encoded_header(raw_header("reference")))

    def test_infinity_and_nan_at_the_end_are_not_hidden_by_finite_prefix(self):
        for code in (0x7F80, 0xFF80, 0x7FC1):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "nonfinite"):
                self.parse(payload=self.payload[:-2] + struct.pack("<H", code))


class SuccessfulProtocolContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capsule = json.loads((ROOT / "review/qwen-block-context-evidence-20261016.json").read_text())
        cls.failed = capsule["block_runs"][-1]["report"]

    def record(self):
        # Exercise the success schema with SYNTHETIC metadata around an existing
        # archived call layout. This does not promote or rewrite that failed run.
        document = copy.deepcopy(self.failed)
        document.update(status="measured", native_returncode=0, raw_artifacts={})
        native = document["native"]
        native.update(status="measured", errors=[], raw_ffn_artifacts_written=True)
        artifacts = {}
        for key, source in (("input", "exposed_ffn_input_sha256"), ("reference", "full_ffn_reference_sha256")):
            value = {"shape": [1, 1024, 4096], "dtype": "BF16", "recipe": success.RECIPE,
                     "payload_bytes": success.PAYLOAD_BYTES, "payload_sha256": native[source],
                     "file_bytes": success.PAYLOAD_BYTES + 256,
                     "file_sha256": hashlib.sha256(("synthetic-" + key).encode()).hexdigest(), "all_finite": True}
            artifacts[key] = value
            document["raw_artifacts"]["ffn-" + key + ".safetensors"] = {
                "bytes": value["file_bytes"], "sha256": value["file_sha256"]}
        document["comparisons"] = success.comparisons(native["samples"], native["measured_groups"])
        return document, artifacts

    def test_all_calls_and_exact_joins_do_not_grant_request_or_full_replay(self):
        document, artifacts = self.record()
        proof = success.verify(document, artifacts)
        self.assertEqual((proof["complete_block_calls"], proof["exact_component_joins"]), (165, 99))
        self.assertTrue(proof["raw_pair_locally_retained"])
        for key in ("all_nine_block_arguments_persisted", "whole_request_qualified",
                    "production_qualified", "physical_overlap_qualified"):
            self.assertFalse(proof[key])
        self.assertFalse(block_context.verify(self.failed)["qualified_performance_ratio_available"])
        self.assertIsNone(self.failed["comparisons"])

    def test_complete_balanced_first_warm_and_measured_order_is_required(self):
        for repeats in (10, 30):
            order = success.expected_order(repeats)
            self.assertEqual(len(order), 5 * (repeats + 3))
            for trial in range(repeats):
                self.assertEqual({arm for phase, row, arm in order if phase == "measured" and row == trial}, set(block_context.ARMS))
        for repeats in (True, 0, 11):
            with self.assertRaises(ValueError):
                success.expected_order(repeats)
        for kind in ("missing", "order", "boolean-time"):
            document, artifacts = self.record()
            samples = document["native"]["samples"]
            if kind == "missing":
                samples.pop()
            elif kind == "order":
                samples[:5] = samples[:5][::-1]
            else:
                samples[0]["wall_ns"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_raw_payload_file_binding_and_complete_pair_are_required(self):
        for kind in ("payload", "file", "shape", "finite", "pair", "written"):
            document, artifacts = self.record()
            if kind == "payload":
                artifacts["input"]["payload_sha256"] = "0" * 64
            elif kind == "file":
                document["raw_artifacts"]["ffn-reference.safetensors"]["sha256"] = "0" * 64
            elif kind == "shape":
                artifacts["input"]["shape"] = [1, 512, 4096]
            elif kind == "finite":
                artifacts["reference"]["all_finite"] = False
            elif kind == "pair":
                artifacts.pop("reference")
            else:
                document["native"]["raw_ffn_artifacts_written"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_fixed_budgets_whole_block_and_prefix_oracles_are_required(self):
        for kind in ("budget", "original", "staged", "prefix", "fp64", "zero-control"):
            document, artifacts = self.record()
            native = document["native"]
            if kind == "budget":
                native["ffn_gate"]["relative_l2_max"] = .1
            elif kind == "original":
                native["samples"][0]["metrics"]["whole_ordinary_block"]["max_abs"] = .01
            elif kind == "staged":
                native["staged_gpu_replay_gate"]["max_abs"] = .01
            elif kind == "zero-control":
                native["runtime_argument_controls"]["zero_private_yields_original_tail"]["max_abs"] = .01
            else:
                sample = next(s for s in native["samples"] if s["arm"] == "shared-hybrid")
                sample["metrics"]["prefix" if kind == "prefix" else "independent_full_ffn"]["relative_l2"] = .031
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_sources_workload_and_original_context_identities_are_required(self):
        for kind in ("source", "inventory", "seed", "prompt", "argument", "share"):
            document, artifacts = self.record()
            if kind == "source":
                document["source_inputs_unchanged"] = False
            elif kind == "inventory":
                document["native_sources_verified_after"] -= 1
            elif kind == "seed":
                document["seed"] = document["baseline_seed"]
            elif kind == "prompt":
                document["prompt_sha256"] = document["baseline_prompt_sha256"]
            elif kind == "argument":
                document["native"]["ordinary_argument_sha256"][0] = "invalid"
            else:
                document["native"]["share"] = 1024
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_alias_copy_accounting_actual_join_and_owner_cleanup_are_required(self):
        for kind in ("alias", "bytes", "join", "release", "unload", "borrowed"):
            document, artifacts = self.record()
            native = document["native"]
            sample = next(s for s in native["samples"] if s["arm"] == "shared-hybrid")
            if kind == "alias":
                sample["identity_after"]["metal_output_matches"] = False
            elif kind == "bytes":
                sample["external_feature_copy_bytes"] = 16777216
            elif kind == "join":
                sample["exact_runtime_join"]["exact_mismatches"] = 1
            elif kind == "release":
                native["restore_adoption_releases"] = 1
            else:
                native["private_unloaded" if kind == "unload" else "borrowed_inputs_outputs_unchanged"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_complete_host_dependencies_are_not_physical_device_overlap(self):
        for kind in ("clock", "join", "copy", "serial"):
            document, artifacts = self.record()
            arm = "copy-hybrid" if kind == "copy" else "shared-serial" if kind == "serial" else "shared-hybrid"
            sample = next(s for s in document["native"]["samples"] if s["arm"] == arm)
            if kind == "clock":
                sample["host_events"][0]["clock_domain"] = "gpu-clock"
            else:
                name = {"join": "gpu-ffn-restore-join-ready", "copy": "cpu-input-feature-copy",
                        "serial": "gpu-remainder-ready"}[kind]
                next(e for e in sample["host_events"] if e["name"] == name)["start_ns"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_fixed_pair_arithmetic_separates_gpu_baseline_from_copy_control(self):
        document, artifacts = self.record()
        samples = document["native"]["samples"]
        comparisons = document["comparisons"]
        self.assertEqual(len(comparisons), 6)
        self.assertEqual(comparisons, success.comparisons(samples, 30))
        gpu_shared, copy_shared = comparisons[3], comparisons[5]
        self.assertEqual((gpu_shared["baseline"], copy_shared["baseline"]), ("gpu-original", "copy-hybrid"))
        for comparison in (gpu_shared, copy_shared):
            expected = []
            for trial in range(30):
                row = {s["arm"]: s["wall_ns"] for s in samples if s["phase"] == "measured" and s["trial"] == trial}
                expected.append(row[comparison["baseline"]] / row[comparison["candidate"]])
            self.assertEqual(comparison["paired_ratios"], expected)
            self.assertEqual(comparison["median_paired_host_ratio"], statistics.median(expected))
        document["comparisons"][3]["median_paired_host_ratio"] += .1
        with self.assertRaises(ValueError):
            success.verify(document, artifacts)

    def test_component_record_cannot_be_promoted_to_product_or_full_request(self):
        for kind in ("production", "physical", "internal-copy", "image", "steps", "zero-load"):
            document, artifacts = self.record()
            if kind == "production":
                document["production_qualified"] = True
            elif kind == "physical":
                document["native"]["physical_overlap_qualified"] = True
            elif kind == "internal-copy":
                document["runtime_internal_copy_behavior"] = "none"
            elif kind == "image":
                document["native"]["image_export"] = True
            elif kind == "steps":
                document["native"]["completed_full_dit_steps"] = 20
            else:
                document["native"]["load_wall_ns"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                success.verify(document, artifacts)

    def test_raw_reference_help_needs_no_model_numpy_or_device(self):
        child = subprocess.run(
            [sys.executable, "-E", "-S", "-B", str(ROOT / "integrations/qwen_block_raw_reference.py"), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("--checkpoint", child.stdout)
        self.assertIn("--bank-run", child.stdout)
        self.assertIn("CPU-only", child.stdout)


if __name__ == "__main__":
    unittest.main()
