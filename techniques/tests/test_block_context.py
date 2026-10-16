"""Offline block evidence, synthetic archive and tiny portable CPU oracle tests."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
sys.path.insert(0, str(ROOT / "integrations"))
import block_context
import qwen_block_shared as wrapper
from capture_qwen_holdout import different_workload


class BlockEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "review/qwen-block-context-evidence-20261016.json").read_text())

    def record(self):
        return copy.deepcopy(self.data["block_runs"][-1]["report"])

    def test_all_partial_outcomes_remain_failed_without_qualified_ratios(self):
        self.assertEqual(len(self.data["block_runs"]), 3)
        calls = joins = 0
        for row in self.data["block_runs"]:
            proof = block_context.verify(row["report"])
            self.assertEqual(proof, row["independent_validation"])
            self.assertTrue(row["owned_wrapper_exit_confirmed"])
            self.assertTrue(row["original_raw_model_source_verified"])
            self.assertFalse(proof["qualified_performance_ratio_available"])
            self.assertIsNone(row["report"]["comparisons"])
            calls += proof["candidate_timed_blocks"]
            joins += proof.get("exact_component_joins", 0)
        self.assertEqual((calls, joins), (330, 198))
        self.assertEqual(len(self.data["capture_admission_rejections"]), 1)
        capture = self.data["capture_admission_rejections"][0]
        self.assertFalse(capture["independent_validation"]["native_inference_started"])
        self.assertEqual(capture["report"]["records"], [])
        for key in ("production_qualified", "qualified_performance_ratios_available", "raw_tensors_available",
                    "raw_addresses_paths_prompts_exported", "data_verifier_reexecuted_original_inputs_or_private_outputs"):
            self.assertFalse(self.data[key])
        for key, relative in (("contract_sha256", "labs/block_context.py"),
                              ("curator_sha256", "scripts/curate_block_context.py")):
            self.assertEqual(self.data[key], hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())

    def test_same_actual_context_and_loop_are_not_a_new_full_request(self):
        rows = self.data["block_runs"]
        self.assertEqual(len({row["timed_loop_and_validation_source_sha256"] for row in rows}), 1)
        for key in ("ordinary_argument_sha256", "ordinary_block_output_sha256", "exposed_ffn_input_sha256", "full_ffn_reference_sha256"):
            self.assertEqual(len({json.dumps(row["report"]["native"][key], sort_keys=True) for row in rows}), 1)
        for row in rows:
            report = row["report"]
            self.assertNotEqual(report["seed"], report["baseline_seed"])
            self.assertNotEqual(report["prompt_sha256"], report["baseline_prompt_sha256"])
            self.assertEqual(report["source_sha256"]["ffn_cpu_witness.hpp"],
                             hashlib.sha256((ROOT / "labs/ffn_cpu_witness.hpp").read_bytes()).hexdigest())
        for row in rows[1:]:
            self.assertFalse(row["report"]["native"]["vae_decode"])
            self.assertFalse(row["report"]["native"]["image_export"])
            self.assertFalse(row["independent_validation"]["original_full_request_candidate_executed"])

    def test_nil_error_is_neither_success_nor_hardware_unsupported(self):
        report = self.data["block_runs"][0]["report"]
        self.assertEqual(report["native"]["private_error"], {"domain": "unknown", "code": 0})
        self.assertFalse(report["native"]["compile_selector_called"])
        self.assertIsNone(report["native"]["compile_wall_ns"])
        proof = block_context.verify(report)
        self.assertEqual(proof["candidate_timed_blocks"], 0)
        self.assertFalse(proof["cause_qualified"])
        self.assertFalse(proof["hardware_unsupported"])

    def test_fixed_whole_block_ffn_and_original_callable_gates_are_required(self):
        for kind in ("block-budget", "ffn-budget", "original", "staged", "prefix", "fp64", "zero-control"):
            report = self.record()
            native = report["native"]
            if kind == "block-budget":
                native["block_gate"]["relative_l2_max"] = .03
            elif kind == "ffn-budget":
                native["ffn_gate"]["relative_l2_max"] = .1
            elif kind == "original":
                next(s for s in native["samples"] if s["arm"] == "gpu-original")["metrics"]["whole_ordinary_block"]["max_abs"] = .01
            elif kind == "staged":
                native["staged_gpu_replay_gate"]["max_abs"] = .01
            elif kind == "zero-control":
                native["runtime_argument_controls"]["zero_ffn_yields_original_residual"]["max_abs"] = .01
            else:
                key = "prefix" if kind == "prefix" else "independent_full_ffn"
                next(s for s in native["samples"] if s["arm"] == "shared-hybrid")["metrics"][key]["relative_l2"] = .031
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                block_context.verify(report)

    def test_missing_or_reordered_calls_and_promoted_failures_are_rejected(self):
        for kind in ("missing", "order", "status", "comparisons", "seed", "prompt", "export"):
            report = self.record()
            if kind == "missing":
                report["native"]["samples"].pop()
            elif kind == "order":
                report["native"]["samples"][:5] = report["native"]["samples"][:5][::-1]
            elif kind == "status":
                report["status"] = "measured"
            elif kind == "comparisons":
                report["comparisons"] = []
            elif kind == "seed":
                report["seed"] = report["baseline_seed"]
            elif kind == "prompt":
                report["prompt_sha256"] = report["baseline_prompt_sha256"]
            else:
                report["native"]["errors"] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                block_context.verify(report)

    def test_copy_accounting_alias_and_normal_cleanup_are_required(self):
        for kind in ("bytes", "count", "alias", "release", "unload", "mapping", "borrowed"):
            report = self.record()
            native = report["native"]
            sample = next(s for s in native["samples"] if s["arm"] == "shared-hybrid")
            if kind == "bytes":
                sample["external_feature_copy_bytes"] = 16777216
            elif kind == "count":
                sample["external_feature_copy_count"] = 2
            elif kind == "alias":
                sample["identity_after"]["metal_output_matches"] = False
            elif kind == "release":
                native["restore_adoption_releases"] = 1
            else:
                native[{"unload": "private_unloaded", "mapping": "owned_mapping_release_pass", "borrowed": "borrowed_inputs_outputs_unchanged"}[kind]] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                block_context.verify(report)

    def test_exact_full_component_join_cannot_be_replaced_by_a_metric(self):
        for kind in ("mismatch", "hash", "elements"):
            report = self.record()
            sample = next(s for s in report["native"]["samples"] if s["arm"] == "shared-hybrid")
            exact = sample["exact_runtime_join"]
            if kind == "mismatch":
                exact["exact_mismatches"] = 1
            elif kind == "hash":
                exact["actual_bf16_sha256"] = "0" * 64
            else:
                exact["elements"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                block_context.verify(report)

    def test_host_dependencies_do_not_grant_physical_overlap(self):
        for kind in ("clock", "duration", "finish", "physical"):
            report = self.record()
            sample = next(s for s in report["native"]["samples"] if s["arm"] == "shared-hybrid")
            if kind == "clock":
                sample["host_events"][0]["clock_domain"] = "gpu-clock"
            elif kind == "duration":
                sample["host_events"][0]["duration_ns"] = True
            elif kind == "finish":
                next(e for e in sample["host_events"] if e["name"] == "modulation-residual-output-ready")["start_ns"] = 0
            else:
                sample["physical_overlap_qualified"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                block_context.verify(report)

    def test_holdout_requires_both_a_new_seed_and_prompt(self):
        different_workload(43, "synthetic different prompt", 42, "synthetic baseline prompt")
        for seed, prompt in ((42, "different"), (43, "baseline"), (True, "different"), (-1, "different"), (2**32, "different"), (43, "\0")):
            with self.subTest(seed=seed, prompt=prompt), self.assertRaises(ValueError):
                different_workload(seed, prompt, 42, "baseline")


class SyntheticArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        count = 4194304
        original = b"\0\x80\x7f\xff\xff\x7f\x80\0" * (count // 4)
        reference = original[::-1]
        planes = original[1::2] + original[::2] + reference[1::2] + reference[::2]
        cls.payload = zlib.compress(planes)
        cls.header = {"schema": "tc-bf16-planes-zlib-v1", "dtype": "BF16", "shape": [1, 1024, 4096],
                      "tensor_order": ["input", "reference"], "plane_order": "high-then-low-per-tensor",
                      "uncompressed_bytes": len(planes), "compressed_bytes": len(cls.payload),
                      "input_sha256": hashlib.sha256(original).hexdigest(), "reference_sha256": hashlib.sha256(reference).hexdigest()}

    def decode(self, header=None, payload=None):
        header = self.header if header is None else header
        payload = self.payload if payload is None else payload
        encoded = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.tcbf16z"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            return wrapper.read_archive(path)

    def test_synthetic_bit_plane_roundtrip_including_nonfinite_bit_patterns(self):
        # Bytes, not a numerical or real-model artifact qualification.
        self.assertEqual(self.decode(), self.header)

    def test_wrong_dtype_shape_length_and_hash_are_rejected(self):
        for key, value in (("dtype", "FP16"), ("shape", [1024, 4096]), ("compressed_bytes", 0), ("input_sha256", "0" * 64)):
            header = dict(self.header)
            header[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.decode(header)

    def test_trailing_or_concatenated_stream_is_rejected(self):
        for extra in (b"trailing", zlib.compress(b"second stream")):
            payload = self.payload + extra
            header = dict(self.header, compressed_bytes=len(payload))
            with self.subTest(extra=extra[:8]), self.assertRaises(ValueError):
                self.decode(header, payload)

    def test_truncated_stream_and_decompression_cap_are_rejected(self):
        for payload in (self.payload[:-1], zlib.compress(b"\0" * 16777217)):
            header = dict(self.header, compressed_bytes=len(payload))
            with self.assertRaises(ValueError):
                self.decode(header, payload)

    def test_short_and_oversized_header_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.tcbf16z"
            for data in (b"short", struct.pack("<Q", 65537)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    wrapper.read_archive(path)


class PortableWitness(unittest.TestCase):
    def test_scalar_cpp_full_and_prefix_match_independent_fsum(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("portable C++20 compiler not installed")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "witness"
            subprocess.run([compiler, "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tests/ffn_cpu_witness_check.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=30)
            child = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
        actual = json.loads(child.stdout)
        inputs = [(i % 9 - 4) / 8 for i in range(16)]
        weights = [(i % 7 - 3) / 16 for i in range(40)]
        down = [(i % 5 - 2) / 8 for i in range(20)]
        for name, width in (("full", 5), ("prefix", 2)):
            expected = []
            for row in range(4):
                hidden = []
                for column in range(width):
                    gate = math.fsum(inputs[row * 4 + k] * weights[column * 4 + k] for k in range(4))
                    up = math.fsum(inputs[row * 4 + k] * weights[(5 + column) * 4 + k] for k in range(4))
                    hidden.append(gate / (1 + math.exp(-gate)) * up)
                expected.extend(math.fsum(hidden[j] * down[column * 5 + j] for j in range(width)) for column in range(4))
            self.assertEqual(len(actual[name]), 16)
            for observed, reference in zip(actual[name], expected):
                self.assertAlmostEqual(observed, reference, delta=1e-18)


if __name__ == "__main__":
    unittest.main()
