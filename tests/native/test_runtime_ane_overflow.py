import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
with mock.patch.object(sys, "path", [str(ROOT/"tools/validation"), *sys.path]):
    from runtime_ane_common import validate_overflow_events, validate_results, validate_gpu_layer_policy


def event(layer=0, begin=0, completed=True):
    return dict(layer=layer, rows=1056, runtime_call_begin=begin, runtime_call_count=3,
                retries=2, headroom_before=1., headroom_after=16., completed=completed)


def receipt(events=None):
    return dict(overflow_events=[] if events is None else events,
                overflow_events_dropped_session_total=0, overflow_retries_session_total=0 if events is None else 2*len(events),
                overflow_event_scope="aggregated per-FFN launch host telemetry; no chunk/physical-engine trace")


class OverflowTests(unittest.TestCase):
    def test_gpu_layer_policy_requires_unsplit_blocks_and_canonical_ordinals(self):
        data=dict(requested_gpu_layers=[2,5],forced_gpu_blocks_session_total=8,
            gpu_blocks_session_total=8,unsplit_gpu_blocks_session_total=8)
        self.assertEqual(validate_gpu_layer_policy(data),((2,5),8))
        self.assertIsNone(validate_gpu_layer_policy({}))
        for key,value in (("requested_gpu_layers",[2,2]),("requested_gpu_layers",[5,2]),
                          ("requested_gpu_layers",[True]),("requested_gpu_layers",[128]),
                          ("forced_gpu_blocks_session_total",9),("unsplit_gpu_blocks_session_total",7)):
            bad={**data,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_gpu_layer_policy(bad)
    def test_empty_legacy_completed_and_failed_launches(self):
        self.assertIsNone(validate_overflow_events({}, 0))
        self.assertEqual(validate_overflow_events(receipt(), 0), ([], 0))
        for completed in (True, False):
            values=receipt([event(completed=completed)])
            self.assertEqual(validate_overflow_events(values, 3), (values["overflow_events"], 0))

    def test_calls_rows_headroom_retry_and_scope_are_not_optional(self):
        changes=[("layer",-1),("rows",0),("retries",0),("runtime_call_count",2),
                 ("runtime_call_begin",1),("headroom_before",None),("headroom_before",True),
                 ("headroom_after",float("nan")),("headroom_after",1.),("completed",1)]
        for key,value in changes:
            bad=receipt([event()]);bad["overflow_events"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_overflow_events(bad,3)
        for key,value in (("overflow_retries_session_total",1),("overflow_events",None),
                          ("overflow_event_scope","physical ANE trace"),("overflow_events_dropped_session_total",True)):
            bad=receipt([event()]);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_overflow_events(bad,3)

    def test_bounded_prefix_and_no_call_aliases(self):
        events=[event(i,3*i) for i in range(32)];values=receipt(events)
        values.update(overflow_events_dropped_session_total=5,overflow_retries_session_total=74)
        self.assertEqual(validate_overflow_events(values,111),(events,5))
        for count in (31,33):
            bad=copy.deepcopy(values);bad["overflow_events"]=[event(i,3*i) for i in range(count)]
            with self.assertRaises(ValueError):validate_overflow_events(bad,111)
        bad=copy.deepcopy(values);bad["overflow_events"][1]["runtime_call_begin"]=2
        with self.assertRaises(ValueError):validate_overflow_events(bad,111)

    def test_native_auto_gpu_decline_cannot_hide_missing_prefix(self):
        # Delegates to the existing fake native calibration fixture rather
        # than loading any SDK, model or device in this host verifier test.
        from tests.native.test_runtime_ane_calibration import report
        calibration=report();calibration.update(status="gpu_only",selected_channels=0,proposed_channels=0,
            trial_passed=False,trial=None,predicted_layer_seconds=.016)
        row=dict(runtime_backend="mlx_cpp_metal",timings_seconds={"request_wall":1.,"denoise":.8},
            hybrid=dict(runtime_failed=False,runtime_failures_session_total=0,runtime_calls_session_total=0,
                runtime_weight=dict(channel_calibration=calibration,executor_backend=None,gpu_blocks_session_total=256,
                    fallback_blocks_session_total=0,device_io_calls_session_total=0,**receipt())))
        validate_results([row],"runtime",1,runtime_backend="private",channel_auto=True)
        later=copy.deepcopy(row);later["hybrid"]["runtime_weight"]["gpu_blocks_session_total"]=512
        del later["hybrid"]["runtime_weight"]["overflow_events"]
        with self.assertRaises(ValueError):validate_results([row,later],"runtime",2,runtime_backend="private",channel_auto=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
