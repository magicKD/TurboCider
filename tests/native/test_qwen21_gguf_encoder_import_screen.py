import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_gguf_encoder_import_screen import validate_record


class EncoderImportScreenTests(unittest.TestCase):
    def test_actual_full_proof_and_shared_buffer(self):
        row = dict(schema="tc-qwen21-encoder-import-screen-v1", qualification_passed=False, source_sha256="a" * 64,
                   workers=4, embedding_mode="packed", verification_bytes=5027785568, source_read_bytes=4500000000,
                   packed_capacity_bytes=5000000000, read_buffer_capacity_bytes=1 << 20,
                   managed_peak_bytes=5000000000 + (1 << 20), final_weight_logical_bytes=5000000000,
                   mlx_logical_peak_bytes=5000000000)
        for field in ("native_verification_seconds", "bank_load_seconds", "bank_read_seconds", "bank_decode_seconds",
                      "embedding_setup_seconds", "token_gather_seconds", "total_observed_seconds"):
            row[field] = .1
        validate_record(row, 4, "packed", "a" * 64, 5027785568)
        for changes in ({"workers": True}, {"workers": 8}, {"verification_bytes": 0},
                        {"verification_bytes": 1}, {"read_buffer_capacity_bytes": 4 << 20},
                        {"managed_peak_bytes": 6000000000}, {"embedding_mode": "dense"},
                        {"bank_decode_seconds": float("nan")}, {"source_sha256": "b" * 64},
                        {"qualification_passed": True}, {"source_read_bytes": False}):
            invalid = copy.deepcopy(row)
            invalid.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_record(invalid, 4, "packed", "a" * 64, 5027785568)


if __name__ == "__main__":
    unittest.main()
