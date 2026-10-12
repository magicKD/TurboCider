from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_private_partition_evidence import validate_grid
from z_private_partition_sweep import candidates
from z_mflux_framework_screen import PRIVATE_SHARES


def fixture():
    grid = candidates(PRIVATE_SHARES, ["f32", "f16"])
    return dict(status="complete_diagnostic", candidates=grid,
        trials=[dict(index=index, **case, exit_code=0, accepted=True) for index, case in enumerate(grid)])


class ZPartitionEvidenceTests(unittest.TestCase):
    def test_terminal_grid_not_incomplete_plan(self):
        self.assertEqual(len(validate_grid(fixture())), 14)
        for key, value in (("status", "running"), ("trials", []), ("candidates", [])):
            value_data = fixture()
            value_data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_grid(value_data)

    def test_failed_candidates_are_retained_not_promoted(self):
        data = fixture()
        data["status"] = "complete_grid_with_retained_failures"
        data["trials"][0].update(exit_code=1, accepted=False)
        validate_grid(data)
        data["trials"][0]["accepted"] = True
        with self.assertRaises(ValueError):
            validate_grid(data)


if __name__ == "__main__":
    unittest.main()
