import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_restore_assets import ranged_prefix, reconstruct_shard_digest


class Flux4AssetReuseTests(unittest.TestCase):
    def test_whole_original_shard_digest_uses_exact_header_and_local_payload(self):
        header = {"weight": dict(dtype="BF16", shape=[2], data_offsets=[0, 4])}
        with tempfile.TemporaryDirectory(prefix="tc-flux4-reuse-") as raw:
            path = Path(raw) / "source"
            path.write_bytes(b"localhdrABCD")
            with path.open("rb") as source:
                digest, names = reconstruct_shard_digest(header, b"localhdr", source.fileno(), header, b"remotehdr", 13)
            self.assertEqual(digest, hashlib.sha256(b"remotehdrABCD").hexdigest())
            self.assertEqual(names, ["weight"])

    def test_wrong_tensor_descriptor_or_noncontiguous_original_is_rejected(self):
        local = {"weight": dict(dtype="BF16", shape=[2], data_offsets=[0, 4])}
        for remote in ({"weight": dict(dtype="F32", shape=[1], data_offsets=[0, 4])},
                       {"weight": dict(dtype="BF16", shape=[2], data_offsets=[1, 5])},
                       {"different": dict(dtype="BF16", shape=[2], data_offsets=[0, 4])}):
            with self.assertRaisesRegex(ValueError, "exact complete original HF"):
                reconstruct_shard_digest(local, b"", -1, remote, b"", 4)

    def test_truncated_local_or_wrong_total_extent_never_proves_reuse(self):
        header = {"weight": dict(dtype="BF16", shape=[2], data_offsets=[0, 4])}
        with tempfile.TemporaryDirectory(prefix="tc-flux4-extent-") as raw:
            path = Path(raw) / "source"
            path.write_bytes(b"AB")
            with path.open("rb") as source:
                with self.assertRaisesRegex(ValueError, "truncated"):
                    reconstruct_shard_digest(header, b"", source.fileno(), header, b"", 4)
            path.write_bytes(b"ABCD")
            with path.open("rb") as source:
                with self.assertRaisesRegex(ValueError, "complete canonical"):
                    reconstruct_shard_digest(header, b"", source.fileno(), header, b"", 5)

    def test_unbounded_remote_header_rejected_before_network(self):
        for count in (0, (1 << 20) + 1):
            with self.assertRaisesRegex(ValueError, "bounded remote header"):
                ranged_prefix("unused", count)


if __name__ == "__main__":
    unittest.main()
