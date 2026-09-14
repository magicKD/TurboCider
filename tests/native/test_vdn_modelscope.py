import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/h3/download_vdn_modelscope.py"


class VdnModelScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("download_vdn_modelscope", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.module = module

    def test_download_contract_is_modelscope_only_and_minimal(self):
        source = SCRIPT.read_text()
        self.assertIn('MODEL_ID = "OpenVDN/vdn-minimax-h3"', source)
        self.assertIn('ENDPOINT = "https://modelscope.cn"', source)
        self.assertNotIn("huggingface_hub", source)
        self.assertNotIn("hf-mirror", source)
        self.assertNotIn("snapshot_download", source)
        self.assertIn(
            "stage-dmd-step-250/linear_branch/model.safetensors",
            self.module.FILES,
        )
        self.assertIn(
            "stage-dmd-step-250/adapters/turbo/adapter_model.safetensors",
            self.module.FILES,
        )
        self.assertFalse(any(key.startswith("stage-b-") for key in self.module.FILES))

    def test_pinned_files_and_hashes(self):
        files = self.module.FILES
        self.assertEqual(len(files), 6)
        self.assertEqual(
            files["stage-dmd-step-250/linear_branch/model.safetensors"]["bytes"],
            4_279_428_112,
        )
        self.assertEqual(
            files["stage-dmd-step-250/linear_branch/model.safetensors"]["sha256"],
            "dec6981c7874f5b3bc92d1a02e256b673a3b3499dc1a124714bb3b19da602855",
        )
        self.assertEqual(
            files["stage-dmd-step-250/adapters/turbo/adapter_model.safetensors"]["bytes"],
            851_452_696,
        )

    def test_revision_and_stage_are_pinned(self):
        self.assertEqual(self.module.MODEL_ID, "OpenVDN/vdn-minimax-h3")
        self.assertEqual(self.module.REVISION, "master")
        self.assertEqual(self.module.STAGE, "stage-dmd-step-250")

    def test_branch_contract_is_50_blocks_of_16_bf16_tensors(self):
        self.assertEqual(len(self.module.BRANCH_TENSORS), 16)
        self.assertEqual(
            self.module.BRANCH_TENSORS["attn.linear_attention.alpha.down.weight"],
            [128, 5376],
        )
        self.assertEqual(
            self.module.BRANCH_TENSORS["attn.to_out_linear.weight"],
            [5376, 7168],
        )

    def test_safetensors_header_reader_rejects_overlapping_records(self):
        header = {
            "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "b": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]},
        }
        raw = json.dumps(header, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0")
            with self.assertRaisesRegex(RuntimeError, "overlapping safetensors"):
                self.module.read_safetensors_header(path)

    def test_adapter_shape_contract_covers_rank16_and_rank64_targets(self):
        self.assertEqual(
            self.module.adapter_module_shapes("norm_out.linear"),
            ([16, 2688], [10752, 16]),
        )
        self.assertEqual(
            self.module.adapter_module_shapes(
                "transformer_blocks.0.adaln_proj.linear"
            ),
            ([16, 2688], [96768, 16]),
        )
        self.assertEqual(
            self.module.adapter_module_shapes(
                "transformer_blocks.0.attn.orig.to_q"
            ),
            ([64, 5376], [7168, 64]),
        )


if __name__ == "__main__":
    unittest.main()
