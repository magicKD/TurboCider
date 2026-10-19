"""CPU-only paired dense fusion qualification and exact full-block contracts."""
from collections import Counter
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import dual_gup_collect as screen
import dual_gup_kernel_collect as kernel
import dual_gup_kernel_audit as oracle
import dual_gup_audit as audit
from test_qwen_base_handoff import report_fixture
from test_qwen_base_profiling import fixture


def real_report(repeats=8):
    r=report_fixture();r.update(neighbor_share_screen=True,gpu_complement_strategy="dual-register",dual_gup=dict(shared=False,bm=32,bn=64,first=0,count=6144,input_dtype="bfloat16",weight_copies=0))
    r["metrics"]["runtime_calls_session_total"]=2*repeats+12;r["metrics"]["runtime_weight"]["hybrid_blocks_session_total"]=2*repeats+12
    for name in ("a","b"):
        r["dual_hidden_quality_"+name]=dict(relative_l2=.0001,elements=1024*6144)
    for name in ("a","b","w"):r["dual_incremental_quality_"+name]=dict(relative_l2=.00001,elements=4194304)
    r["repeats"]=repeats
    r["samples"]=[dict(arm=screen.ARMS[(j+rep+4)%4],rep=rep,warmup=rep<0,input_variant=bool(rep&1),
        seconds=.025 if screen.ARMS[(j+rep+4)%4]=="private-original" else .026,
        **(dict(completion_deltas=dict(prediction_wait_seconds=.01,output_wait_seconds=.0003,caller_wait_seconds=.011)) if (j+rep+4)%4>=2 else {}))
        for rep in range(-2,repeats) for j in range(4)]
    r["gpu_complement_alone_samples"]=[dict(arm=("original","fused")[(j+rep+2)%2],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.01)
        for rep in range(-2,repeats) for j in range(2)]
    return r


def observation(r):
    raw,_=fixture();template=next(e for e in raw["events"] if e["name"]=="ane.evaluate_api")
    raw["events"]=[e for e in raw["events"] if e["request"]==-2]
    scopes={1,2,3,4,5,101,102,104}|{6+i for i,s in enumerate(r["samples"]) if s["arm"].startswith("private-")}
    raw["events"] += [dict(template,request=n,model=1) for n in sorted(scopes)]
    raw["counts"]=dict(Counter(e["name"] for e in raw["events"]))
    return raw


class DualGupTests(unittest.TestCase):
    def test_synthetic_full_stage_contract_is_bits_not_only_numeric_zero(self):
        cases=[]
        for i,(m,k,f,first,count) in enumerate(((33,128,320,17,257),(1024,4096,12288,0,6144))):
            for shared in (False,True):
                for stage in range(5):
                    q=dict(bits_exact=True,changed=0,relative_l2=0.,max_abs=0.,elements=m*count)
                    cases.append(dict(fixture=i,m=m,k=k,f=f,first=first,count=count,shared=shared,stage=stage,bm=32,bn=64,quality_vs_mlx=q.copy(),quality_vs_mpp=q.copy()))
        report=dict(status="complete",synthetic_only=True,complete_request=False,image_export=False,extra_packed_weight_bytes=0,rejected=20,
                    source_unchanged=True,owned_output_restoration_exact=True,cases=cases)
        kernel.validate(report)
        for change in (lambda x:x["cases"].pop(),lambda x:x["cases"][0]["quality_vs_mlx"].update(bits_exact=False),
                       lambda x:x["cases"][0].update(first=0),lambda x:x.update(source_unchanged=False)):
            bad=copy.deepcopy(report);change(bad)
            with self.assertRaises(ValueError):kernel.validate(bad)

    def test_periodic_projection_oracle_matches_direct_full_k_dots(self):
        m,k,f,first,n=3,32,19,2,7
        x=oracle.expected_input(m,k)[0]
        ids=(np.arange(2*f)[:,None]+2)*k+np.arange(k)[None,:]
        w=oracle.bf16(((((ids%257)*37+13)%31)-15)/128.)
        for changed_input,changed_weight in ((False,False),(True,False),(False,True)):
            xx=oracle.bf16(x*.75+.015625) if changed_input else x
            ww=oracle.bf16(w*.875+.0009765625) if changed_weight else w
            got=oracle.projections(m,k,f,first,n,changed_input,changed_weight)
            for actual,offset in zip(got,(first,f+first)):
                wanted=oracle.bf16(xx.astype(np.float64)@ww[offset:offset+n].astype(np.float64).T)
                np.testing.assert_array_equal(actual[0],wanted)

    def test_full_block_counts_and_primary_denominator_are_matched_hybrid(self):
        r=real_report();a=screen.validate(r,8,6144,False)
        self.assertEqual(a["native_workload_predictions"],28)
        self.assertAlmostEqual(a["incremental_reduction"],1-.026/.025)
        screen.validate(real_report(16),16,6144,False)
        for change in (lambda x:x["metrics"].update(runtime_calls_session_total=15),
                       lambda x:x["metrics"]["runtime_weight"].update(fallback_blocks_session_total=1),
                       lambda x:x.update(gpu_packed_weight_bytes=1024),lambda x:x["dual_gup"].update(shared=True),
                       lambda x:x.update(complete_request=True)):
            bad=copy.deepcopy(r);change(bad)
            with self.assertRaises(ValueError):screen.validate(bad,8,6144,False)

    def test_cannot_drop_reference_calls_or_extra_observed_prediction(self):
        r=real_report();raw=observation(r);a=screen.observer_contract(raw,r,8)
        self.assertEqual(a["actual_workload_evaluations"],28)
        for change in (lambda x:x["events"].pop(),lambda x:x["events"].append(dict(x["events"][-1],request=2000)),
                       lambda x:x["events"][-1].update(model=2),lambda x:x["events"][-1].update(name="ane.load_api")):
            bad=copy.deepcopy(raw);change(bad);bad["counts"]=dict(Counter(e["name"] for e in bad["events"]))
            with self.assertRaises(ValueError):screen.observer_contract(bad,r,8)

    def test_sample_order_warmup_completion_and_separate_gpu_controls_are_strict(self):
        for change in (lambda x:x["samples"].reverse(),lambda x:x["samples"][0].update(warmup=False),
            lambda x:x["gpu_complement_alone_samples"].pop(),lambda x:x["gpu_complement_alone_samples"][0].update(seconds=float("nan")),
            lambda x:next(s for s in x["samples"] if s["arm"]=="private-original")["completion_deltas"].update(output_wait_seconds=-1),
            lambda x:x["dual_hidden_quality_a"].update(relative_l2=.005),lambda x:x["dual_incremental_quality_w"].update(relative_l2=.003)):
            bad=real_report();change(bad)
            with self.assertRaises(ValueError):screen.validate(bad,8,6144,False)

    def test_full_audit_keeps_source_original_hybrid_and_same_recipe_replay_exact(self):
        records=[]
        for case,kind,mode in (("generate","register","stock"),("generate","shared","stock"),("generate","register","timeline"),("edit","register","stock")):
            records.append(dict(cohort=str(len(records)),case=case,kind=kind,channels=6144,mode=mode,
                block_sha256={**dict.fromkeys(("ffn_input","block_reference","alternate_reference","other_weight_reference","private_original_a","private_original_b","private_original_w"),case),"block_candidate":case+kind},
                gpu_sha256={**dict.fromkeys(("input_a","input_b","hidden_original_a","hidden_original_b","partial_original_a","partial_original_b"),case),"hidden_fused_a":case+kind},
                analysis=dict(median_seconds={"private-original":.024,"private-fused":.0239},incremental_reduction=1-.0239/.024,gpu_complement_alone_medians={})))
        value=audit.matched(records);self.assertIsNone(value["comparisons"][2]["incremental_reduction"])
        for change in (lambda x:x[1]["block_sha256"].update(private_original_a="other"),
            lambda x:x[1]["gpu_sha256"].update(input_b="other"),lambda x:x[2]["block_sha256"].update(block_candidate="other")):
            bad=copy.deepcopy(records);change(bad)
            with self.assertRaises(ValueError):audit.matched(bad)


if __name__=="__main__":unittest.main()
