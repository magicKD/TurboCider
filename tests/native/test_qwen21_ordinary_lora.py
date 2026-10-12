"""CPU binding regression for ordinary Qwen21 LoRA checkpoint geometry."""

import json
import os
import platform
import struct
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class OrdinaryLoRABindingTests(unittest.TestCase):
    def test_all_default_suffix_pairs_bind_or_fail_closed(self):
        mlx_root = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        native = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native"))
        self.assertTrue((native / "libturbocider.dylib").is_file())
        with tempfile.TemporaryDirectory(prefix="turbocider-qwen21-lora-") as directory:
            binary = Path(directory) / "probe"
            subprocess.run([
                "xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra",
                "-mmacosx-version-min=" + platform.mac_ver()[0],
                str(ROOT / "tools/native/qwen21_lora_binding_probe.cpp"),
                "-I", str(ROOT / "native/core"), "-isystem", str(mlx_root / "include"),
                "-L" + str(native), "-lturbocider", "-L" + str(mlx_root / "lib"), "-lmlx",
                "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx_root / "lib"),
                "-o", str(binary),
            ], check=True, cwd=ROOT)
            result = subprocess.run([str(binary), directory], capture_output=True,
                                    text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["projections"], 224)
            self.assertTrue(report["unknown_rejected"])
            self.assertTrue(report["unsupported_tensor_rejected"])
            self.assertTrue(report["orphan_alpha_rejected"])
            self.assertTrue(report["oversized_b_rejected"])
            self.assertTrue(report["zero_rank_rejected"])
            self.assertTrue(report["incomplete_rejected"])
            self.assertTrue(report["legacy_subset_preserved"])

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_QWEN21_ORDINARY_LORA"),
                         "set TURBOCIDER_TEST_QWEN21_ORDINARY_LORA for installed header audit")
    def test_installed_ordinary_adapter_has_224_complete_pairs(self):
        path = Path(os.environ["TURBOCIDER_TEST_QWEN21_ORDINARY_LORA"])
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            self.assertLess(header_size, 1024 * 1024)
            header = json.loads(stream.read(header_size))
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        self.assertEqual(len(tensors), 448)
        targets = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                   "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"}
        expected = {f"transformer_blocks.{block}.{target}" for block in range(32) for target in targets}
        self.assertEqual({name.removesuffix(".lora_A.default.weight") for name in tensors
                          if name.endswith(".lora_A.default.weight")}, expected)
        self.assertEqual({name.removesuffix(".lora_B.default.weight") for name in tensors
                          if name.endswith(".lora_B.default.weight")}, expected)
        for stem in expected:
            a, b = tensors[stem + ".lora_A.default.weight"], tensors[stem + ".lora_B.default.weight"]
            self.assertEqual(a["dtype"], "BF16")
            self.assertEqual(b["dtype"], "BF16")
            self.assertEqual(a["shape"][0], 16)
            self.assertEqual(b["shape"][1], 16)


if __name__ == "__main__":
    unittest.main()
