"""Bad measurement/component options reject before loading a GPU library."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tools/native/run_z_image_gguf_quantized.py"


class MeasurementOptionsTests(unittest.TestCase):
    def reject(self, options, message, env=None):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-probe-options-") as raw:
            output = Path(raw) / "output"
            result = subprocess.run([sys.executable, str(PROBE), "--library", "missing.dylib", "--model", raw,
                "--output", str(output), *options], capture_output=True, text=True, env=env, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(message, result.stderr)
            self.assertFalse(output.exists())

    def test_diagnostics_cannot_pollute_timing_or_memory(self):
        for mode in ("timing", "memory"):
            for flag in (["--dump"], ["--cancel-once-at-block", "0"], ["--cancel-once-at-encoder-layer", "0"],
                         ["--cancel-once-at-refiner", "0"]):
                self.reject(["--measurement", mode, *flag], "requires diagnostic")

    def test_invalid_warmup_and_prefetch(self):
        for value in ("-1", "9"):
            self.reject(["--warmup", value], "warmup 0..8")
        self.reject(["--prefetch", "3"], "prefetch -1=native packed")

    def test_baseline_and_dense_bf16_controls_are_separate(self):
        self.reject(["--baseline-bf16"], "BF16 baseline requires prefetch -1")
        self.reject(["--baseline-bf16","--prefetch","-1","--native-import","cpu_direct"], "BF16 baseline requires")
        self.reject(["--precision","z-dense-bf16-v1"], "requires bounded packed_streamed")
        self.reject(["--precision","z-dense-bf16-v1","--source-residency","packed_streamed","--prefetch","-1"], "requires bounded packed_streamed")
        self.reject(["--precision","z-dense-bf16-v1","--source-residency","packed_streamed","--gpu-eval-blocks"], "requires bounded packed_streamed")
        self.reject(["--alternate-prompt","different"], "alternate prompt requires diagnostic")
        self.reject(["--alternate-prompt","different","--prefetch","-1","--native-import","cpu_direct",
                     "--retain-packed","--runs","3","--measurement","timing"], "alternate prompt requires diagnostic")

    def test_component_inputs_are_atomic_and_explicit(self):
        self.reject(["--encoder-gguf", "missing.gguf"], "must be supplied together")
        self.reject(["--encoder-gguf", "missing.gguf", "--encoder-config", "missing.json",
                     "--encoder-tokenizer", "missing.json"], "missing encoder")
        self.reject(["--encoder-weight-limit-bytes", "0"], "must be positive")

    def test_direct_import_requires_explicit_compatible_selector(self):
        self.reject(["--native-import","cpu_direct"], "requires only native packed")
        self.reject(["--native-weight-limit-bytes","1024"], "requires CPU-direct")
        self.reject(["--native-import","cpu_direct","--prefetch","-1","--native-weight-limit-bytes","0"],
                    "positive native weight ceiling")
        self.reject(["--cancel-once-at-import-tensor","0"], "requires CPU-direct")
        self.reject(["--cancel-once-at-block","0","--cancel-once-at-refiner","0"], "mutually exclusive")
        self.reject(["--compile-packed"], "packed compile requires CPU-direct")
        self.reject(["--retain-packed"], "packed retention requires CPU-direct")
        self.reject(["--compile-packed","--native-import","cpu_direct","--prefetch","-1","--gpu-eval-blocks"],
                    "packed compile requires CPU-direct")

    def test_tokenizer_mismatch_and_environment_conflict(self):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-probe-component-") as raw:
            root = Path(raw); (root / "tokenizer").mkdir()
            native = root / "tokenizer/tokenizer.json"; native.write_text("native")
            other = root / "other.json"; other.write_text("other")
            config = root / "config.json"; config.write_text("{}")
            checkpoint = root / "encoder.gguf"; checkpoint.write_bytes(b"fake fixture; never loaded")
            common = [sys.executable, str(PROBE), "--library", "missing.dylib", "--model", str(root),
                "--output", str(root / "output"), "--encoder-gguf", str(checkpoint), "--encoder-config", str(config)]
            result = subprocess.run([*common, "--encoder-tokenizer", str(other)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2); self.assertIn("tokenizer differs", result.stderr)
            env = dict(os.environ, TURBOCIDER_Z_QWEN3_GGUF="unrelated.gguf")
            result = subprocess.run([*common, "--encoder-tokenizer", str(native)], capture_output=True, text=True, env=env, timeout=10)
            self.assertEqual(result.returncode, 2); self.assertIn("conflicting encoder environment", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__": unittest.main()
