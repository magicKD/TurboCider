"""CPU-only checks for the isolated LTX 0.9.8 resident hybrid experiment."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
import ltx098_ane_hybrid_resident_abba as probe


class ResidentValidationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(seed=42, chunks=3, step_mask=192)

    def detail(self, mode="hybrid", request=0):
        hybrid = mode == "hybrid"
        return {"mode": mode, "seed": 42 + request, "request_index": request, "requests_in_process": 2,
                "actual_steps": 8, "ffn_calls": 224 if hybrid else 0,
                "hybrid_ffn_calls": 56 if hybrid else 0, "coreml_calls": 168 if hybrid else 0,
                "text_cache_hit": bool(request), "cross_attention_cache_hit": bool(request),
                "hybrid_step_mask": 192, "per_block_calls": [8] * 28 if hybrid else [0] * 28,
                "per_step_ffn_calls": [28] * 8 if hybrid else [0] * 8,
                "per_step_hybrid_ffn_calls": [0] * 6 + [28, 28] if hybrid else [0] * 8}

    def test_accepts_exact_cold_and_warm_contracts(self):
        for mode, expected in (("gpu", 0), ("hybrid", 56)):
            for request in (0, 1):
                with self.subTest(mode=mode, request=request):
                    self.assertEqual(probe.validate_detail(self.detail(mode, request), mode, request, self.args), expected)

    def test_rejects_cache_counts_and_step_mismatches(self):
        for update in ({"text_cache_hit": False}, {"seed": 42}, {"coreml_calls": 167},
                       {"per_block_calls": [8] * 27 + [7]},
                       {"per_step_ffn_calls": [28] * 7 + [27]},
                       {"per_step_hybrid_ffn_calls": [0] * 5 + [28, 28, 0]}):
            with self.subTest(update=update):
                detail = self.detail(request=1)
                detail.update(update)
                with self.assertRaises(ValueError):
                    probe.validate_detail(detail, "hybrid", 1, self.args)

    def test_graph_receipt_checks_content_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graph.mlmodelc"
            graph.mkdir()
            payload = graph / "weights.bin"
            payload.write_bytes(b"fixed graph")
            manifest = root / "manifest.json"
            spec = {"backend": "ltx098_experimental_bias_fp16", "biases": True, "rows": 256,
                    "hidden": 2048, "width": 8192, "files": {"graph.mlmodelc/weights.bin": probe.sha256(payload)}}
            manifest.write_text(json.dumps(spec))
            self.assertEqual(probe.graph_receipt(manifest), graph)
            payload.write_bytes(b"altered graph")
            with self.assertRaises(ValueError):
                probe.graph_receipt(manifest)
            payload.write_bytes(b"fixed graph")
            (graph / "extra").symlink_to(payload)
            with self.assertRaises(ValueError):
                probe.graph_receipt(manifest)


if __name__ == "__main__":
    unittest.main()
