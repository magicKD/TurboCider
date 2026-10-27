"""CPU controls for truthful fixed-program receipts and selector checks."""
import copy
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import collect
import fixed_program_collect as screen
import fallback_collect


def runtime_receipt():
    return dict(fixed_program_requested=True,fixed_program_enabled=True,program_disabled=False,
        program_instances_session_total=1,blocked_rebuilds_session_total=0,overflow_retries_session_total=0,
        headroom_scale=1,fallback_blocks_session_total=0)


def negative_fixture():
    failed={**runtime_receipt(),"program_disabled":True,"blocked_rebuilds_session_total":1,"fallback_blocks_session_total":1}
    metrics=dict(runtime_weight=failed,runtime_calls_session_total=1)
    after=dict(fixed=True,instances=1,blocked_rebuilds=1,disabled=True)
    report=dict(status="complete",synthetic_only=True,complete_request=False,strict=True,image_export=False,
        rows=32,hidden=128,ane_channels=512,full_width=1024,
        direct=dict(captured_policy_survived_env_change=True,old_output_unchanged=True,
            safe=dict(calls=1),overflow=dict(ok=False,calls=1,overflow_retries=0),
            after=after,after_repeat=after,repeat_stage_rejected=True,repeat=dict(calls=0)),
        hybrid=dict(source_unchanged=True,disabled_peer_no_model=True,complete_gpu_bits_exact=True,sticky_failure=True,
            first_retained_resources=False,full_gpu_calls=2,first_metrics=metrics,second_metrics=metrics))
    def event(name,phase):
        return dict(name=name,request=phase,begin_ns=100,end_ns=110,ok=True,model=1,queue=1,thread=1,gpu_begin_ns=0,gpu_end_ns=0)
    calibration=event("metal.command_buffer",-2);calibration.update(gpu_begin_ns=101,gpu_end_ns=109)
    rows=[calibration]+[event("ane.evaluate_api",phase) for phase,count in {0:3,1:1,2:1,4:3,5:1}.items() for _ in range(count)]
    observed=dict(enabled=True,drained=True,restored_imps=True,pending=0,dropped=0,events=rows,
                  counts=dict(Counter(r["name"] for r in rows)))
    return report,observed


class FixedProgramProfileTests(unittest.TestCase):
    def test_exactly_one_policy_switch_in_matched_environment(self):
        off=collect.environment("runtime","stock",6144,True,False,True)
        on=collect.environment("runtime","stock",6144,True,False,True,True)
        self.assertEqual(set(off),set(on))
        self.assertEqual({k for k in off if off[k]!=on[k]},{"TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM"})
        for route,mode in (("gpu","stock"),("cache","stock"),("runtime","runtime-layers")):
            with self.assertRaises(ValueError):collect.environment(route,mode,4096,False,False,False,True)

    def test_receipt_rejects_silent_recreation_fallback_and_disabled_owner(self):
        collect.validate_fixed_program([dict(hybrid=dict(runtime_weight=runtime_receipt()))])
        for key,value in (("fixed_program_requested",False),("fixed_program_enabled",False),("program_disabled",True),
            ("program_instances_session_total",2),("program_instances_session_total",True),("blocked_rebuilds_session_total",1),
            ("overflow_retries_session_total",1),("headroom_scale",4),("fallback_blocks_session_total",1)):
            receipt=runtime_receipt();receipt[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                collect.validate_fixed_program([dict(hybrid=dict(runtime_weight=receipt))])

    def test_real_selector_contract_cannot_be_replaced_by_native_zero_counters(self):
        report,observed=negative_fixture()
        self.assertTrue(screen.validate(report,observed,True)["hot_no_compile"])
        row=copy.deepcopy(observed["events"][-1]);row["name"]="ane.compile_api"
        observed["events"].append(row);observed["counts"]["ane.compile_api"]=1
        with self.assertRaises(ValueError):screen.validate(report,observed,True)

    def test_subsequent_fallback_must_not_submit_ane_or_allocate_disabled_peer(self):
        for phase in (6,7):
            report,observed=negative_fixture();row=copy.deepcopy(observed["events"][-1]);row["request"]=phase
            observed["events"].append(row);observed["counts"]["ane.evaluate_api"]+=1
            with self.subTest(phase=phase),self.assertRaises(ValueError):screen.validate(report,observed,True)

    def test_whole_session_failure_must_retain_both_owners_counters(self):
        rows=[]
        for i,seed in enumerate((73,74,73)):
            rows.append(dict(metrics=dict(model="qwen-image-2.1",operation="image.generate",width=512,height=512,
                steps=40,actual_denoise_steps=40,seed=seed,backend="mlx_cpp_metal",
                hybrid=dict(runtime_failed=True,runtime_failures_session_total=1,runtime_calls_session_total=0,
                    runtime_weight=dict(fixed_program_requested=True,fixed_program_enabled=False,program_instances_session_total=0,
                        blocked_rebuilds_session_total=0,gpu_blocks_session_total=1280*(i+1),hybrid_blocks_session_total=0,
                        overflow_retries_session_total=0,failure_reason="W8 graph/banks exceed memory budget")),
                qwen_private_qkv=dict(failed=True,predictions_session_total=0,slot_bytes=0,gpu_blocks_session_total=1280*(i+1),
                    failure_reason="fixed FFN disabled; complete GPU request"))))
        fallback_collect.validate(rows)
        for change in (lambda r:r[1]["metrics"]["hybrid"]["runtime_weight"].update(gpu_blocks_session_total=1280),
            lambda r:r[1]["metrics"]["qwen_private_qkv"].update(gpu_blocks_session_total=1280),
            lambda r:r[0]["metrics"]["hybrid"]["runtime_weight"].update(program_instances_session_total=1),
            lambda r:r[0]["metrics"].update(backend="private_ane")):
            value=copy.deepcopy(rows);change(value)
            with self.assertRaises(ValueError):fallback_collect.validate(value)


if __name__=="__main__":unittest.main()
