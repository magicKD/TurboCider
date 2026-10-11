from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_private_partition_sweep import ORDER, candidates
from z_mflux_framework_screen import PRIVATE_SHARES


class ZPartitionSweepTests(unittest.TestCase):
    def test_complete_declared_grid_keeps_join_recipes_separate(self):
        rows = candidates(PRIVATE_SHARES, ["f32", "f16"])
        self.assertEqual(len(rows), 14)
        self.assertEqual(rows[0], dict(channels=2048, join="f32"))
        self.assertEqual(rows[-1], dict(channels=8192, join="f16"))
        self.assertEqual(ORDER.split(","), list(reversed(ORDER.split(","))))
        self.assertEqual({name: ORDER.split(",").count(name) for name in set(ORDER.split(","))},
            {"native-gpu": 2, "native-frozen": 2, "native-private": 2})

    def test_unknown_duplicate_or_empty_candidates_are_rejected(self):
        for shares, joins in (([], ["f32"]), ([4096], []), ([1234], ["f32"]), ([4096,4096], ["f32"]),
                ([4096], ["f32","f32"]), ([4096], ["unknown"])):
            with self.subTest(shares=shares, joins=joins), self.assertRaises(ValueError):
                candidates(shares, joins)


if __name__ == "__main__":
    unittest.main()
