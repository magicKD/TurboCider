import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
SPEC=importlib.util.spec_from_file_location("convrot_mpp_screen",ROOT/"tools/validation/convrot_partial_mpp_screen.py")
SCREEN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SCREEN)


class ConvRotPartialScreenTests(unittest.TestCase):
    def rows(self,mode="mpp"):
        return [dict(steps=4,actual_denoise_steps=4,timings_seconds=dict(request_wall=7.0),
            acceleration_selection=SCREEN.MARKER if mode=="mpp" else "original",
            runtime_backend="mlx_cpp_metal_convrot+private_ane_runtime_weight_experimental",
            hybrid=dict(runtime_failed=False,runtime_calls_session_total=128*(index+1),runtime_weight=dict(
                executor_backend="private_ane",data_path="w8a8_convrot",partition_axis="intermediate_channels",
                ane_channels=4096,fp32_channel_join_enabled=True,fallback_blocks_session_total=0,
                overflow_retries_session_total=0,channel_blocks_session_total=128*(index+1)))) for index in range(3)]

    def test_real_completed_channel_work_required_for_selection(self):
        SCREEN.validate_rows(self.rows(),"mpp",4);SCREEN.validate_rows(self.rows("original"),"original",4)
        for field,value in (("channel_blocks_session_total",0),("channel_blocks_session_total",True),
                ("fallback_blocks_session_total",1),("overflow_retries_session_total",1),
                ("ane_channels",0),("executor_backend","public_coreml"),("fp32_channel_join_enabled",False)):
            rows=self.rows();rows[1]["hybrid"]["runtime_weight"][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):SCREEN.validate_rows(rows,"mpp",4)
        rows=self.rows();rows[1]["hybrid"]["runtime_calls_session_total"]=128
        with self.assertRaises(ValueError):SCREEN.validate_rows(rows,"mpp",4)
        rows=self.rows();rows[0]["hybrid"]["runtime_failed"]=True
        with self.assertRaises(ValueError):SCREEN.validate_rows(rows,"mpp",4)
        with self.assertRaises(ValueError):SCREEN.validate_rows(self.rows("original"),"mpp",4)
        with self.assertRaises(ValueError):SCREEN.validate_rows(self.rows(),"original",4)

    def test_gpu_control_cannot_be_split_or_missing_steps(self):
        row=dict(steps=4,actual_denoise_steps=4,timings_seconds=dict(request_wall=4.0),acceleration_selection="gpu",
            runtime_backend="mlx_cpp_metal_convrot_compiled_experimental")
        SCREEN.validate_rows([copy.deepcopy(row) for _ in range(3)],"gpu",4)
        for key,value in (("actual_denoise_steps",3),("runtime_backend","mlx_cpp_metal_convrot_packed_q8"),
                ("hybrid",dict(runtime_calls_session_total=1)),("acceleration_selection",SCREEN.MARKER)):
            rows=[copy.deepcopy(row) for _ in range(3)];rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):SCREEN.validate_rows(rows,"gpu",4)

    def test_explicit_cold_retry_diagnostic_does_not_allow_warm_retry_or_missing_events(self):
        rows=self.rows()
        for row in rows:
            runtime=row["hybrid"]["runtime_weight"]
            runtime.update(overflow_retries_session_total=2,headroom_scale=16,
                overflow_event_scope="aggregated per-FFN launch host telemetry; no chunk/physical-engine trace",
                overflow_events_dropped_session_total=0,overflow_events=[dict(completed=True,layer=2,rows=1056,
                    runtime_call_begin=4,runtime_call_count=3,retries=2,headroom_before=1,headroom_after=16)])
        with self.assertRaises(ValueError):SCREEN.validate_rows(rows,"mpp",4)
        SCREEN.validate_rows(rows,"mpp",4,cold_retry_cap=2)
        for field,value in (("headroom_scale",4),("overflow_retries_session_total",3),("overflow_events",[])):
            bad=copy.deepcopy(rows);bad[1]["hybrid"]["runtime_weight"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):SCREEN.validate_rows(bad,"mpp",4,cold_retry_cap=2)


if __name__=="__main__":unittest.main()
