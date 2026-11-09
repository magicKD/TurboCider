"""The two distilled checkpoints have distinct noise and timestep schedules."""
import os
from pathlib import Path
import subprocess
import unittest

from tools.validation.wan22_schedule import warped_sigmas


PROBE = Path(os.environ.get("TURBOCIDER_WAN22_SCHEDULE_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native Wan2.2 schedule probe required")
class Wan22QuanhaolScheduleTests(unittest.TestCase):
    def test_shift8_and_bfloat16_timestep_narrowing(self):
        import torch

        rows = [[float(value) for value in line.split()] for line in
                subprocess.check_output([str(PROBE)], text=True).splitlines()]
        self.assertEqual(len(rows), 7)
        fast = warped_sigmas()
        four = warped_sigmas((1000, 750, 500, 250), shift=8)
        for actual, expected in zip(rows[:3], fast):
            self.assertAlmostEqual(actual[0], expected, places=6)
            self.assertAlmostEqual(actual[1], expected * 1000, places=3)
        expected_times = (torch.tensor(four) * 1000).to(torch.bfloat16).to(torch.int64).tolist()
        for actual, sigma, timestep in zip(rows[3:], four, expected_times):
            self.assertAlmostEqual(actual[0], sigma, places=6)
            self.assertEqual(actual[1], timestep)


if __name__ == "__main__":
    unittest.main()
