import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "experimental/transformer/scripts"),
)
from evaluate_vae_tiling import TENSOR_NAMES, compare_tensors


class VAETilingInputTests(unittest.TestCase):
    def create_tensors(self, root: Path, payload: bytes) -> None:
        directory = root / "request-0-tensors"
        directory.mkdir(parents=True)
        for name in TENSOR_NAMES:
            (directory / f"{name}.bf16").write_bytes(payload)

    def test_tensor_comparison_requires_exact_nonempty_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = root / "reference", root / "candidate"
            self.create_tensors(reference, b"\x00\x01")
            self.create_tensors(candidate, b"\x00\x01")
            result = compare_tensors(reference, candidate, 0)
            self.assertEqual(set(result), set(TENSOR_NAMES))
            self.assertTrue(all(item["byte_exact"] for item in result.values()))

            (candidate / "request-0-tensors/stage2_video.bf16").write_bytes(
                b"\x00\x02")
            with self.assertRaises(ValueError):
                compare_tensors(reference, candidate, 0)

    def test_tensor_comparison_rejects_empty_or_mismatched_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = root / "reference", root / "candidate"
            self.create_tensors(reference, b"\x00\x01")
            self.create_tensors(candidate, b"\x00\x01")
            (candidate / "request-0-tensors/stage1_video.bf16").write_bytes(b"")
            with self.assertRaises(ValueError):
                compare_tensors(reference, candidate, 0)
