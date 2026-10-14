from copy import deepcopy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_vae_phase_summary import expected_counts,validate_trace,summarize


class PhaseDiagnosticGuards(unittest.TestCase):
    def fixture(self):
        rows=[]
        for key,calls in expected_counts(1).items():
            for _ in range(calls):
                rows.append(dict(schema="tc_seedvr2_vae_phase_profile_v1",index=len(rows),key=key,next_output_frame=0,
                    inclusive=key.endswith("inclusive"),initial=True,seconds=.001,input_shape=[1,1,64,64,128],
                    output_shape=[1,1,64,64,128],active_before=10,active_after=20,peak_before=20,
                    peak_after=20,cache_after=0,history_unique_before=5,history_unique_after=5,
                    history_logical_after=10,history_unavailable_after=0))
        rows[0]["peak_before"]=10;rows[0]["peak_after"]=20
        rows[1]["peak_before"]=20;rows[1]["peak_after"]=30
        for row in rows[2:]:row["peak_before"]=row["peak_after"]=30
        observation=dict(frames=1,route="baseline",quality_passed=True,bitwise_equal=True,
            diagnostic_phase_count=len(rows),mlx_active_peak_bytes=30)
        return rows,observation

    def test_original_partition_counts_and_complete_trace(self):
        self.assertEqual(sum(expected_counts(9).values()),167)
        self.assertEqual(sum(expected_counts(25).values()),471)
        validate_trace(*self.fixture())
        for frames in (0,2,True):
            with self.assertRaises(ValueError):expected_counts(frames)

    def test_missing_frames_bad_history_or_peak_are_rejected(self):
        for change in ({"schema":"old"},{"history_unique_after":21},{"peak_after":1},
                       {"next_output_frame":1},{"history_unavailable_after":1},{"seconds":float("nan")}):
            rows,observation=self.fixture();rows[-1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_trace(rows,observation)
        rows,observation=self.fixture()
        with self.assertRaises(ValueError):validate_trace(rows[:-1],observation)
        for change in ({"bitwise_equal":False},{"mlx_active_peak_bytes":31},{"route":"spatial"}):
            with self.assertRaises(ValueError):validate_trace(rows,{**observation,**change})

    def test_inclusive_scopes_are_not_added_twice(self):
        rows,observation=self.fixture();validate_trace(rows,observation)
        report=summarize(rows)
        self.assertAlmostEqual(report["noninclusive_phase_seconds"],sum(row["seconds"] for row in rows if not row["inclusive"]))
        self.assertEqual(report["highest_peak_region"]["peak_after"],30)
        self.assertFalse(report["highest_peak_region"]["inclusive"])


if __name__=="__main__":unittest.main()
