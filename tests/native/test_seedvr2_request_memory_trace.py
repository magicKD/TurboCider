from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_request_memory_trace import summarize


class RequestMemoryTraceGuards(unittest.TestCase):
    def rows(self):
        return [dict(phase="encode",active_bytes=10,peak_bytes=20,cache_bytes=2),
            dict(phase="dit",active_bytes=30,peak_bytes=40,cache_bytes=3),
            dict(phase="decode",active_bytes=25,peak_bytes=40,cache_bytes=4)]
    def test_first_attainment_is_an_interval_not_a_later_phase_maximum(self):
        value=summarize(self.rows(),40)
        self.assertEqual(value["first_observed_peak_event"]["phase"],"dit")
        self.assertEqual(value["preceding_event"]["phase"],"encode")
    def test_missing_shared_counters_nonmonotonic_or_invalid_values_are_rejected(self):
        for peak in (0,False,True,39,41):
            with self.subTest(peak=peak),self.assertRaises(ValueError):summarize(self.rows(),peak)
        for changes in ({"peak_bytes":19},{"active_bytes":41},{"cache_bytes":-1},{"peak_bytes":float("nan")}):
            rows=self.rows();rows[-1].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):summarize(rows,40)
    def test_observer_does_not_mutate_allocator_or_force_gpu_work(self):
        source=(ROOT/"tools/validation/seedvr2_request_memory_trace.py").read_text()
        for forbidden in ("mx.eval(","mx.synchronize(","mx.reset_peak_memory(","mx.clear_cache(","mx.set_cache_limit(","mx.array("):
            self.assertNotIn(forbidden,source)


if __name__=="__main__":unittest.main()
