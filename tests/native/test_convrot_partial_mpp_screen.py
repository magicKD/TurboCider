import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
SPEC=importlib.util.spec_from_file_location("convrot_mpp_screen",ROOT/"tools/validation/convrot_partial_mpp_screen.py")
SCREEN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SCREEN)
STATE_SPEC=importlib.util.spec_from_file_location("convrot_lora_switch",ROOT/"tools/validation/convrot_lora_state_switch.py")
STATE=importlib.util.module_from_spec(STATE_SPEC);STATE_SPEC.loader.exec_module(STATE)


class ConvRotPartialScreenTests(unittest.TestCase):
    def test_shared_mpp_and_bf16_require_same_actual_matched_geometry(self):
        rows=self.rows()
        for row in rows:
            row["text_tokens"]=37;row["hybrid"]["bucket"]=1152
            row["acceleration_selection"]+="; experimental request-matched Private FFN rows=1152"
        SCREEN.validate_rows(rows,"mpp",4,template_rows=1056,matched_rows=True)
        narrow=copy.deepcopy(rows)
        for row in narrow:row["acceleration_selection"]=row["acceleration_selection"].replace(SCREEN.MARKER,SCREEN.BF16_MARKER)
        SCREEN.validate_rows(narrow,"bf16",4,template_rows=1056,matched_rows=True)
        for changes in (("bucket",1056),("runtime_calls_session_total",248)):
            bad=copy.deepcopy(rows);bad[0]["hybrid"][changes[0]]=changes[1]
            with self.assertRaises(ValueError):SCREEN.validate_rows(bad,"mpp",4,template_rows=1056,matched_rows=True)
        bad=copy.deepcopy(rows);bad[0]["acceleration_selection"]=SCREEN.MARKER
        with self.assertRaises(ValueError):SCREEN.validate_rows(bad,"mpp",4,template_rows=1056,matched_rows=True)

    def test_matched_bucket_requires_selected_grid_and_actual_calls(self):
        rows=self.rows("original")
        for row in rows:
            row["hybrid"]["bucket"]=1152
            row["acceleration_selection"]=SCREEN.BF16_MARKER+"; experimental request-matched Private FFN rows=1152"
        SCREEN.validate_rows(rows,"bf16_matched",4)
        for value in (1056,1088):
            bad=copy.deepcopy(rows);bad[0]["hybrid"]["bucket"]=value
            with self.assertRaises(ValueError):SCREEN.validate_rows(bad,"bf16_matched",4)
        bad=copy.deepcopy(rows);bad[0]["acceleration_selection"]=SCREEN.BF16_MARKER
        with self.assertRaises(ValueError):SCREEN.validate_rows(bad,"bf16_matched",4)
        kept=copy.deepcopy(rows)
        for row in kept:
            row["hybrid"]["bucket"]=1056
            row["acceleration_selection"]=SCREEN.BF16_MARKER+"; experimental request-matched Private FFN rows=1056"
        SCREEN.validate_rows(kept,"bf16_matched",4,template_rows=1056)
        with self.assertRaises(ValueError):SCREEN.validate_rows(rows,"bf16_matched",4,template_rows=1056)
    def rows(self,mode="mpp"):
        return [dict(steps=4,actual_denoise_steps=4,width=512,height=512,text_tokens=31,timings_seconds=dict(request_wall=7.0),
            acceleration_selection=SCREEN.MARKER if mode=="mpp" else "original",
            runtime_backend="mlx_cpp_metal_convrot+private_ane_runtime_weight_experimental",
            hybrid=dict(bucket=1056,runtime_failed=False,runtime_calls_session_total=128*(index+1),runtime_weight=dict(
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

    def test_bf16_rounding_is_explicit_and_cannot_relabel_exact_or_gpu(self):
        rows=self.rows("original")
        for row in rows:row["acceleration_selection"]=SCREEN.BF16_MARKER
        SCREEN.validate_rows(rows,"bf16",4)
        for mode in ("original","mpp","gpu"):
            with self.assertRaises(ValueError):SCREEN.validate_rows(rows,mode,4)
        with self.assertRaises(ValueError):SCREEN.validate_rows(self.rows(),"bf16",4)
        mixed=copy.deepcopy(rows);mixed[0]["acceleration_selection"]+="; "+SCREEN.MARKER
        with self.assertRaises(ValueError):SCREEN.validate_rows(mixed,"bf16",4)

    def test_caption_bucket_cliff_requires_real_calls_not_only_successful_blocks(self):
        rows=self.rows("original")
        for index,row in enumerate(rows):
            row.update(text_tokens=37,acceleration_selection=SCREEN.BF16_MARKER)
            row["hybrid"]["runtime_calls_session_total"]=248*(index+1)
        SCREEN.validate_rows(rows,"bf16",4)
        wrong=copy.deepcopy(rows)
        for row in wrong:row["hybrid"]["bucket"]=1152
        with self.assertRaises(ValueError):SCREEN.validate_rows(wrong,"bf16",4)
        for index,row in enumerate(wrong):row["hybrid"]["runtime_calls_session_total"]=128*(index+1)
        SCREEN.validate_rows(wrong,"bf16",4)

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

    def test_real_lora_and_sensitive_policy_cannot_be_forged(self):
        rows=self.rows()
        for index,row in enumerate(rows):
            row.update(lora_strategy="inference_time",lora_applied_projections=238)
            row["acceleration_selection"]+="; "+SCREEN.LORA_MARKER
            r=row["hybrid"]["runtime_weight"]
            r.update(channel_blocks_session_total=124*(index+1),requested_gpu_layers=[2],
                forced_gpu_blocks_session_total=4*(index+1),lora_channel_range_calls_session_total=124*(index+1),
                lora_channel_full_calls_session_total=0)
        SCREEN.validate_rows(rows,"mpp",4,has_lora=True,gpu_layers=(2,))
        for field,value in (("lora_channel_range_calls_session_total",0),("forced_gpu_blocks_session_total",0),
                ("requested_gpu_layers",[]),("lora_channel_full_calls_session_total",1)):
            bad=copy.deepcopy(rows);bad[1]["hybrid"]["runtime_weight"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):SCREEN.validate_rows(bad,"mpp",4,has_lora=True,gpu_layers=(2,))
        bad=copy.deepcopy(rows);bad[0]["lora_strategy"]="in_memory_merge"
        with self.assertRaises(ValueError):SCREEN.validate_rows(bad,"mpp",4,has_lora=True,gpu_layers=(2,))

    def test_same_executor_base_adapter_strength_base_state(self):
        rows=[]
        for index,adapter in enumerate((False,True,True,False)):
            row=self.rows()[0]
            row.update(steps=8,actual_denoise_steps=8,lora_strategy="inference_time" if adapter else "none",lora_applied_projections=238 if adapter else 0)
            if adapter:row["acceleration_selection"]+="; "+SCREEN.LORA_MARKER
            row["hybrid"].update(load_seconds=1.5,runtime_calls_session_total=248*(index+1))
            row["hybrid"]["runtime_weight"].update(channel_blocks_session_total=248*(index+1),requested_gpu_layers=[2],
                forced_gpu_blocks_session_total=8*(index+1),headroom_scale=1,lora_channel_full_calls_session_total=0,
                lora_channel_range_calls_session_total=(0,248,496,496)[index])
            rows.append(row)
        hashes=["base","adapter1","adapter_half","base"]
        STATE.verify(rows,hashes)
        narrow=copy.deepcopy(rows)
        for row in narrow:row["acceleration_selection"]=row["acceleration_selection"].replace(SCREEN.MARKER,SCREEN.BF16_MARKER)
        STATE.verify(narrow,hashes,True)
        with self.assertRaises(ValueError):STATE.verify(narrow,hashes)
        with self.assertRaises(ValueError):STATE.verify(rows,hashes,True)
        for field,value in (("lora_channel_range_calls_session_total",0),("overflow_retries_session_total",1)):
            bad=copy.deepcopy(rows);bad[2]["hybrid"]["runtime_weight"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):STATE.verify(bad,hashes)
        bad=copy.deepcopy(rows);bad[3]["hybrid"]["load_seconds"]=2.0
        with self.assertRaises(ValueError):STATE.verify(bad,hashes)
        with self.assertRaises(ValueError):STATE.verify(rows,["base","adapter1","adapter_half","changed_base"])


if __name__=="__main__":unittest.main()
