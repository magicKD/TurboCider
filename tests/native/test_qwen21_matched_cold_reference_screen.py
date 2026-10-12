import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_matched_cold_reference_screen import file_identity, check_identities, native_env, write_requests


class MatchedColdReferenceTests(unittest.TestCase):
    def test_quick_generation_check_has_no_payload_read(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "weights.bin"; path.write_bytes(b"original")
            identity = {str(path): file_identity(path)}
            with patch.object(Path, "open", side_effect=AssertionError("payload was read")):
                check_identities(identity)
            st = path.stat(); path.write_bytes(b"modified")
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            with self.assertRaises(ValueError): check_identities(identity)

    def test_replace_and_retarget_invalidate_even_same_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "weights.bin"; path.write_bytes(b"original")
            alias = Path(raw) / "alias"; alias.symlink_to(path)
            identities = {str(alias): file_identity(alias)}
            other = Path(raw) / "other"; other.write_bytes(b"original")
            alias.unlink(); alias.symlink_to(other)
            with self.assertRaises(ValueError): check_identities(identities)
            with self.assertRaises(ValueError): file_identity(Path(raw) / "missing")

    def test_actual_precondition_and_target_recipes_are_isolated(self):
        with patch.dict(os.environ, {"TURBOCIDER_QWEN21_GGUF_READ_AHEAD": "invalid", "TURBOCIDER_PRIVATE_ANE_CHANNELS": "9999"}):
            stream = native_env(Path("/private/cache"), streamed=True)
            gpu = native_env(Path("/private/cache"))
            mixed = native_env(Path("/private/cache"), mixed=True)
        self.assertEqual(stream["TURBOCIDER_QWEN21_BF16_STREAMING"], "1")
        self.assertEqual(stream["TURBOCIDER_QWEN21_GGUF_READ_AHEAD"], "0")
        self.assertEqual(stream["TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS"], "0")
        self.assertNotIn("TURBOCIDER_PRIVATE_ANE_CHANNELS", stream)
        self.assertEqual(gpu["TURBOCIDER_QWEN21_GGUF_READ_BUFFER_MIB"], "4")
        self.assertEqual(gpu["TURBOCIDER_QWEN21_GGUF_READ_AHEAD"], "1")
        self.assertEqual(mixed["TURBOCIDER_PRIVATE_ANE_CHANNELS"], "5120")
        self.assertEqual(mixed["TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE"], "all")
        self.assertEqual(stream["TURBOCIDER_QWEN21_SOURCE_REGISTRATION"], "1")

    def test_real_equal_precondition_requests_and_fresh_prompts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = write_requests(root, "pre", 40, streamed=True)
            data = [json.loads(Path(p).read_text()) for p in paths]
            self.assertEqual(len({r["prompt"] for r in data}), 3)
            for row in data:
                self.assertEqual((row["width"], row["height"], row["steps"], row["seed"]), (512, 512, 40, 29))
                self.assertEqual(row["residency"], "component_staged")
                self.assertEqual(row["execution"], "gpu")
                self.assertNotIn("dump_tensors", row)


if __name__ == "__main__":
    unittest.main()
