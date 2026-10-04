"""Runtime-weight graph ABI and opt-in native CPU/NE + MLX integration.

No real checkpoints or generated media. The temporary graph artifacts and
probe executable are scoped to this test and removed on exit.
"""

import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("runtime_ane_export", ROOT / "tools/coreml/export_runtime_ane.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
HAS_COREML = importlib.util.find_spec("coremltools") is not None


@unittest.skipUnless(sys.platform == "darwin" and HAS_COREML and
                     os.environ.get("TURBOCIDER_TEST_RUNTIME_ANE") == "1",
                     "set TURBOCIDER_TEST_RUNTIME_ANE=1 on macOS for tiny Prepared lifecycle checks")
class PreparedGraphIntegrationTests(unittest.TestCase):
    """One tiny CPU-only graph; no MLX/Metal work or real checkpoints."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="tc-prepared-graph-test-")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name)
        cls.report_root = Path(os.environ.get("TURBOCIDER_PREPARED_TEST_REPORT_DIR",
                                             str(cls.root / "evidence")))
        cls.report_root.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(ROOT / "tools/validation"))
        from runtime_ane_memory import run_owned
        cls.run_owned = staticmethod(run_owned)
        env = {**os.environ, "TURBOCIDER_NATIVE_OUT": str(cls.root / "build")}
        with (cls.report_root / "build.stdout.log").open("w") as stdout, \
                (cls.report_root / "build.stderr.log").open("w") as stderr:
            built = run_owned(["bash", "tools/native/build_ane_runtime_probe.sh"],
                              cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=120)
        if built.returncode:
            raise RuntimeError(f"focused Prepared probe build failed; see {cls.report_root}")
        cls.probe = cls.root / "build/ane-runtime-probe"

    def test_prepared_ownership_abi_budget_and_synchronous_equivalence(self):
        sources = ("native/backends/ane_runtime.hpp", "native/backends/ane_runtime.mm",
                   "native/backends/ane_artifact_lease.hpp", "tools/native/ane_runtime_probe.cpp",
                   "tools/coreml/export_runtime_ane.py", "tests/native/test_ane_runtime.py")
        hashes = lambda: {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in sources}
        report = {"passed": False, "scope": "tiny 32x64x96 CPU-only Core ML; no MLX/Metal",
                  "source_sha256_before": hashes(),
                  "probe_sha256": hashlib.sha256(self.probe.read_bytes()).hexdigest()}
        fixture_root = None
        try:
            with tempfile.TemporaryDirectory(prefix="fixture-", dir=self.root) as fixture:
                fixture_root = Path(fixture)
                compiler_tmp = fixture_root / "compiler-temp"
                compiler_tmp.mkdir(mode=0o700)
                private_tmp = fixture_root / "lease-temp"
                private_tmp.mkdir(mode=0o700)
                directory = fixture_root / "graph"
                command = [sys.executable, str(ROOT / "tools/coreml/export_runtime_ane.py"),
                           "--kind", "swiglu", "--rows", "32", "--hidden", "64",
                           "--width", "96", "--tile-k", "33", "--tile-n", "47",
                           "--lora-inputs", "--output", str(directory)]
                with (self.report_root / "export.stdout.log").open("w") as stdout, \
                        (self.report_root / "export.stderr.log").open("w") as stderr:
                    exported = self.run_owned(command, cwd=ROOT,
                        env={**os.environ, "TMPDIR": str(compiler_tmp)},
                        stdout=stdout, stderr=stderr, timeout=120)
                self.assertEqual(exported.returncode, 0, "tiny export failed; inspect retained export logs")
                manifest = directory / "manifest.json"
                metadata = json.loads(manifest.read_text())
                self.assertEqual((metadata["rows"], metadata["hidden"], metadata["width"]), (32, 64, 96))
                self.assertEqual(metadata["graph_version"], 2)
                # Keep the artifact receipt valid and logical FFN geometry
                # unchanged; only the declared row ABI disagrees with MLModel.
                bad_abi = directory / "bad-abi.json"
                bad_abi.write_text(json.dumps({**metadata, "rows": 31}))
                report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
                command = [str(self.probe), str(manifest), "prepared-lifecycle-self-test", str(bad_abi)]
                with (self.report_root / "probe.stdout.log").open("w") as stdout, \
                        (self.report_root / "probe.stderr.log").open("w") as stderr:
                    result = self.run_owned(command, cwd=ROOT,
                        env={**os.environ, "TMPDIR": str(private_tmp)},
                        stdout=stdout, stderr=stderr, timeout=60)
                report["exit_code"] = result.returncode
                report["private_leases_after_exit"] = sorted(
                    path.name for path in private_tmp.glob("turbocider-runtime-ane-*"))
                self.assertEqual(result.returncode, 0, "Prepared lifecycle probe failed; inspect retained probe logs")
                data = json.loads((self.report_root / "probe.stdout.log").read_text())
                report["probe"] = data
                for key in ("passed", "cpu_only", "unbound_cleanup", "move_cleanup",
                            "bind_budget_rejected", "abi_rejected", "source_move_survived",
                            "same_output_bits", "same_snapshot_after_bind"):
                    self.assertIs(data[key], True, key)
                self.assertEqual(data["leases_after_destroy"], 0)
                self.assertEqual(report["private_leases_after_exit"], [])
                self.assertGreater(data["slot_bytes"], 0)
                self.assertGreaterEqual(data["estimated_bytes"], data["slot_bytes"])
                for name in ("synchronous_timings", "prepared_timings"):
                    times = data[name]
                    self.assertTrue(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                                        for value in times.values()), name)
                    self.assertAlmostEqual(times["load"], times["artifact"] + times["model_load"] + times["bind"], places=12)
                self.assertFalse((directory / "graph.mlmodelc").exists())
                self.assertTrue((directory / "graph.moved/model.mil").is_file())
                self.assertEqual(hashes(), report["source_sha256_before"], "sources changed during the probe")
            report["passed"] = True
        except BaseException as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["fixtures_removed"] = fixture_root is not None and not fixture_root.exists()
            report["source_sha256_after"] = hashes()
            (self.report_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


@unittest.skipUnless(HAS_COREML, "coremltools not installed")
class GraphTests(unittest.TestCase):
    def test_lora_is_activation_only_and_requires_swiglu(self):
        spec = EXPORT.geometry("swiglu", 32, 64, 96, 33, 47, lora_inputs=True)
        self.assertEqual(spec["graph_version"], 2)
        function = EXPORT.make_program(spec).functions["main"]
        self.assertEqual(set(function.inputs), {"x", "wg", "wu", "wd", "dg", "du"})
        self.assertEqual({value.name for value in function.outputs}, {"y", "h"})
        for op in function.operations:
            if op.op_type == "const":
                self.assertLess(len(op.outputs[0].shape), 2)
        for kind in ("matmul", "gelu"):
            with self.assertRaisesRegex(ValueError, "SwiGLU"):
                EXPORT.geometry(kind, 32, 64, 96, 33, 47, lora_inputs=True)

    def test_graph_has_runtime_weights_and_internal_tail_tiles(self):
        for kind in ("matmul", "swiglu", "gelu"):
            with self.subTest(kind=kind):
                spec = EXPORT.geometry(kind, 32, 64, 96, 33, 47)
                function = EXPORT.make_program(spec).functions["main"]
                self.assertEqual(set(function.inputs), set(spec["inputs"]))
                operations = list(function.operations)
                self.assertGreater(sum(op.op_type == "matmul" for op in operations), 1)
                self.assertTrue(all(op.inputs["transpose_y"].val for op in operations if op.op_type == "matmul"))
                for op in operations:
                    if op.op_type == "const":
                        self.assertLess(len(op.outputs[0].shape), 2, "matrix weights must be inputs, never constants")


@unittest.skipUnless(sys.platform == "darwin" and HAS_COREML and
                     os.environ.get("TURBOCIDER_TEST_RUNTIME_ANE") == "1",
                     "set TURBOCIDER_TEST_RUNTIME_ANE=1 on macOS for Core ML/MLX integration")
class RuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="tc-runtime-ane-test-")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name)
        env = {**os.environ, "TURBOCIDER_NATIVE_OUT": str(cls.root / "build")}
        subprocess.run(["bash", "tools/native/build_ane_runtime_probe.sh"], cwd=ROOT, env=env,
                       check=True, capture_output=True, text=True, timeout=120)
        cls.probe = cls.root / "build/ane-runtime-probe"
        if (ROOT / "build/native/libturbocider.dylib").is_file():
            subprocess.run(["bash", "tools/native/build_ane_ffn_test.sh"], cwd=ROOT, env=env,
                           check=True, capture_output=True, text=True, timeout=120)

    def run_probe(self, manifest, policy="ne"):
        return subprocess.run([str(self.probe), str(manifest), "2", "97", policy],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def assert_probe_samples(self, data, chunks, repeats=2):
        self.assertEqual(data["warmup_iterations"], 2)
        self.assertEqual(data["measured_iterations"], repeats)
        self.assertEqual(len(data["samples"]), repeats)
        self.assertEqual(data["gpu_rows"] + data["ane_rows"], data["rows"])
        self.assertEqual(data["ane_rows"], chunks * data["chunk"])
        timing_fields = ("gpu_seconds", "split_seconds", "split_with_stage_seconds",
                         "stage_seconds", "predict_seconds", "join_seconds",
                         "input_seconds", "output_seconds", "runtime_seconds")
        for name in timing_fields:
            values = [sample[name] for sample in data["samples"]]
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values), name)
            self.assertAlmostEqual(data[name], statistics.median(values), places=12)
        for key, denominator in (("speedup_without_stage", "split_seconds"),
                                 ("speedup_with_stage", "split_with_stage_seconds")):
            self.assertGreater(data[denominator], 0)
            self.assertAlmostEqual(data[key], data["gpu_seconds"] / data[denominator], places=12)
        for index, sample in enumerate(data["samples"]):
            self.assertEqual(sample["iteration"], index)
            self.assertEqual(sample["order"], "split,gpu" if index % 2 else "gpu,split")
            self.assertEqual(sample["calls"], chunks + sample["overflow_retries"])
            self.assertGreaterEqual(sample["headroom_scale"], 1)
            self.assertGreaterEqual(sample["split_with_stage_seconds"], sample["split_seconds"])
            self.assertGreaterEqual(sample["split_seconds"], sample["join_seconds"])
            # These are disjoint worker intervals, not GPU/ANE latency sums.
            self.assertGreaterEqual(sample["runtime_seconds"] + 1e-9,
                                    sum(sample[key] for key in (
                                        "input_seconds", "predict_seconds", "output_seconds")))

    @unittest.skipUnless((ROOT / "build/native/libturbocider.dylib").is_file(),
                         "build native runtime for model-facing fallback/cancellation tests")
    def test_model_wrapper_recomputes_failed_tail_and_drains_cancellation(self):
        directory = self.root / "model-wrapper"
        EXPORT.export(directory, EXPORT.geometry("swiglu", 32, 64, 96, 33, 47))
        lora_directory = self.root / "model-wrapper-lora"
        EXPORT.export(lora_directory, EXPORT.geometry("swiglu", 32, 64, 96, 33, 47, lora_inputs=True))
        env = {**os.environ, "TURBOCIDER_NATIVE_OUT": str(self.root / "build"),
               "TURBOCIDER_RUNTIME_ANE_CHUNKS": "2"}
        result = subprocess.run([str(self.root / "build/ane-ffn-test"), str(directory / "manifest.json"),
                                 str(lora_directory / "manifest.json")],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU tail fallback", result.stdout)
        self.assertIn("PASS Q4/Q8 staging vs MLX dequantize", result.stdout)
        self.assertIn("PASS runtime LoRA activation corrections", result.stdout)
        self.assertIn("PASS runtime LoRA readiness: lazy upstream input", result.stdout)
        self.assertIn("PASS runtime base v2: validation-only hidden", result.stdout)
        self.assertIn("PASS correction input isolation: CPU/NE", result.stdout)
        self.assertIn("PASS Qwen LoRA correction halves", result.stdout)
        self.assertIn("PASS LoRA alpha: F32/BF16/F16/integer", result.stdout)
        self.assertIn("PASS output lifetime: BF16/FP16/FP32", result.stdout)
        self.assertIn("PASS async head join: base/LoRA", result.stdout)
        self.assertIn("PASS request scheduler isolation: base/A/same-A/B/base", result.stdout)
        self.assertIn("PASS resident memory pressure: graph release and complete GPU result", result.stdout)
        print(result.stdout.strip())

    @unittest.skipUnless((ROOT / "build/native/libturbocider.dylib").is_file(),
                         "build native runtime for Z-Image padding geometry")
    def test_z_gguf_padding_singleton_and_unquantized_contract(self):
        result = subprocess.run([str(self.root / "build/z-image-padding-test")],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS Z-Image GGUF padding", result.stdout)

    @unittest.skipUnless((ROOT / "build/native/libturbocider.dylib").is_file(),
                         "build native runtime for compiled Qwen split switching")
    def test_qwen_compiled_full_split_variants(self):
        result = subprocess.run([str(self.root / "build/qwen21-runtime-split-test")],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS Qwen full/split cache switching", result.stdout)
        print(result.stdout.strip())

    def test_real_predictions_rebind_weights_and_reuse_chunk_backing(self):
        for kind in ("matmul", "swiglu", "gelu"):
            directory = self.root / kind
            EXPORT.export(directory, EXPORT.geometry(kind, 32, 64, 96, 33, 47))
            for policy in ("cpu", "ne"):
                with self.subTest(kind=kind, policy=policy):
                    result = self.run_probe(directory / "manifest.json", policy)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(result.stdout)
                    self.assert_probe_samples(data, 1)
                    self.assertEqual((data["hidden"], data["width"], data["tile_k"], data["tile_n"]),
                                     (64, 96, 33, 47))
                    self.assertFalse(data["lora_inputs"])
                    self.assertTrue(data["cpu_reference_checked"])
                    self.assertEqual(data["ane_rows"], 32)
                    self.assertLess(data["relative_l2"], .03)
                    self.assertGreater(data["cosine"], .999)
            # Explicit two-chunk tail with a non-aligned GPU head.
            result = subprocess.run([str(self.probe), str(directory / "manifest.json"),
                                     "2", "97", "ne", "-", "2"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["ane_rows"], 64)
            self.assert_probe_samples(data, 2)
            before = (directory / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                EXPORT.export(directory, EXPORT.geometry(kind, 32, 64, 96, 33, 47))
            self.assertEqual((directory / "manifest.json").read_bytes(), before)

    def test_matmul_probe_selects_qwen_q_or_z_qkv_with_explicit_weight_priority(self):
        directory = self.root / "attention-source"
        EXPORT.export(directory, EXPORT.geometry("matmul", 32, 64, 96, 33, 47))
        z_qkv = "noise_refiner.0.attention.qkv.weight"
        qwen_q = "transformer_blocks.0.attn.to_q.weight"
        for keys, expected in (((z_qkv,), z_qkv), ((z_qkv, "w"), "w"),
                               ((qwen_q,), qwen_q), ((qwen_q, z_qkv), qwen_q),
                               ((qwen_q, "w"), "w")):
            with self.subTest(expected=expected):
                # The unselected tensor has the wrong projection width, so
                # success proves tensor selection, not merely the receipt.
                # No Python safetensors/torch dependency in this fixture.
                offsets, payload = {}, b""
                for key in keys:
                    width = 96 if key == expected else 95
                    data = (b"\x80\x3f" if key == "w" else b"\x00\x40") * (width * 64)
                    offsets[key] = {"dtype": "BF16", "shape": [width, 64],
                                    "data_offsets": [len(payload), len(payload) + len(data)]}
                    payload += data
                header = json.dumps(offsets, separators=(",", ":")).encode()
                header += b" " * (-len(header) % 8)
                bundle = self.root / f"{expected.replace('.', '-')}.safetensors"
                bundle.write_bytes(struct.pack("<Q", len(header)) + header + payload)
                result = subprocess.run(
                    [str(self.probe), str(directory / "manifest.json"), "2", "97", "cpu",
                     str(bundle), "1", "simd"], cwd=ROOT,
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                receipt = json.loads(result.stdout)
                self.assertEqual(receipt["weight_source"], expected)
                self.assertTrue(receipt["real_weights"])
                self.assert_probe_samples(receipt, 1)

    def test_qwen_qkv_pack_requires_three_real_square_projections(self):
        directory = self.root / "qwen-qkv-pack"
        EXPORT.export(directory, EXPORT.geometry("matmul", 32, 32, 96, 17, 47))
        keys = tuple(f"transformer_blocks.0.attn.to_{projection}.weight"
                     for projection in "qkv")
        for available, mode, succeeds in ((keys, "qkv", True),
                                           (keys, "qkv-separate-gpu", True),
                                           (keys, "qkv-parts", True),
                                           (keys[:-1], "qkv", False)):
            offsets, payload = {}, b""
            for index, key in enumerate(available):
                # Distinct Q/K/V values catch source order and row-offset bugs.
                data = (b"\x00\x3e", b"\x80\x3e", b"\x00\x3f")[index] * (32 * 32)
                offsets[key] = {"dtype": "BF16", "shape": [32, 32],
                                "data_offsets": [len(payload), len(payload) + len(data)]}
                payload += data
            header = json.dumps(offsets, separators=(",", ":")).encode()
            header += b" " * (-len(header) % 8)
            bundle = self.root / f"qkv-{len(available)}.safetensors"
            bundle.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            result = subprocess.run(
                [str(self.probe), str(directory / "manifest.json"), "2", "97", "cpu",
                 str(bundle), "1", "simd", "-", "0", mode], cwd=ROOT,
                capture_output=True, text=True, timeout=120)
            if succeeds:
                self.assertEqual(result.returncode, 0, result.stderr)
                receipt = json.loads(result.stdout)
                self.assertEqual(receipt["projection_pack"],
                                 "parts" if mode == "qkv-parts" else "qkv")
                self.assertEqual(receipt["gpu_projection"],
                                 "separate" if mode != "qkv" else "single")
                self.assertEqual(receipt["weight_source"], "+".join(keys))
                self.assertFalse(receipt["captured_input"])
                self.assert_probe_samples(receipt, 1)
                self.assertTrue(receipt["cpu_reference_checked"])
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(keys[-1], result.stderr)

    def test_artifact_tamper_and_missing_file_rejected_before_prediction(self):
        for tamper in ("changed", "missing", "extra", "symlink"):
            directory = self.root / tamper
            EXPORT.export(directory, EXPORT.geometry("matmul", 32, 64, 96, 33, 47))
            artifact = directory / "graph.mlmodelc/model.mil"
            if tamper == "changed":
                artifact.write_bytes(artifact.read_bytes() + b"\n")
            elif tamper == "missing":
                artifact.unlink()
            elif tamper == "extra":
                (directory / "graph.mlmodelc/unlisted.txt").write_text("unlisted")
            else:
                moved = directory / "original.mil"
                artifact.rename(moved)
                artifact.symlink_to(moved)
            result = self.run_probe(directory / "manifest.json")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("runtime ANE", result.stderr)

    def test_loaded_graph_uses_private_verified_artifact_and_releases_it(self):
        directory = self.root / "leased-graph"
        EXPORT.export(directory, EXPORT.geometry("matmul", 32, 64, 96, 33, 47))
        private_tmp = self.root / "leased-temp"
        private_tmp.mkdir()
        result = subprocess.run(
            [str(self.probe), str(directory / "manifest.json"), "lease-source-move-self-test"],
            cwd=ROOT, env={**os.environ, "TMPDIR": str(private_tmp)},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS private runtime ANE artifact lease", result.stdout)
        self.assertFalse((directory / "graph.mlmodelc").exists())
        self.assertTrue((directory / "graph.moved/model.mil").is_file())
        self.assertEqual(list(private_tmp.iterdir()), [], "private compiled snapshot leaked after process exit")

    def test_v2_chunk_sizes_keep_the_same_total_gpu_and_ane_rows(self):
        # This is geometry/correctness coverage, not a latency assertion.
        # Both graphs perform the same 64-row ANE tail and 33-row GPU head.
        # Exercise both even and odd sample medians without asserting speed.
        for chunk, chunks, repeats in ((16, 4, 2), (32, 2, 3)):
            with self.subTest(chunk=chunk):
                directory = self.root / f"matched-rows-v2-c{chunk}"
                EXPORT.export(directory, EXPORT.geometry(
                    "swiglu", chunk, 64, 96, 33, 47, lora_inputs=True))
                result = subprocess.run(
                    [str(self.probe), str(directory / "manifest.json"),
                     str(repeats), "97", "ne", "-", str(chunks)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assert_probe_samples(data, chunks, repeats)
                self.assertTrue(data["lora_inputs"])
                self.assertTrue(data["cpu_reference_checked"])
                self.assertEqual((data["gpu_rows"], data["ane_rows"]), (33, 64))
                self.assertLess(data["relative_l2"], .03)
                self.assertGreater(data["cosine"], .999)

    def test_overflow_headroom_restores_bf16_output(self):
        import mlx.core as mx

        directory = self.root / "headroom"
        EXPORT.export(directory, EXPORT.geometry("swiglu", 32, 64, 96, 32, 48))
        # Gate=80, up=1600 => hidden product overflows FP16, while the
        # mathematical output remains finite in the BF16 residual dtype.
        # Dense positive inputs/weights avoid cancellation masking overflow.
        x = mx.full((97, 64), 1., mx.bfloat16)
        gate = mx.full((96, 64), 1.25, mx.bfloat16)
        up = mx.full((96, 64), 25., mx.bfloat16)
        down = mx.full((64, 96), .125, mx.bfloat16)
        bundle = directory / "synthetic.safetensors"
        mx.save_safetensors(str(bundle), {"x": x, "wg": gate, "wu": up, "wd": down})
        result = subprocess.run([str(self.probe), str(directory / "manifest.json"), "2", "97",
                                 "ne", str(bundle), "2"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertGreater(data["headroom_scale"], 1)
        self.assertGreater(data["overflow_retries_including_warmup"], 0)
        self.assertLess(data["relative_l2"], .03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
