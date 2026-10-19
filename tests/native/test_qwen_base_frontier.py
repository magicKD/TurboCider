"""CPU-only six-arm frontier counters, output controls and scope negatives."""
from collections import Counter
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import frontier_collect as screen
import frontier_audit as audit
from test_qwen_base_handoff import report_fixture
from test_qwen_base_profiling import fixture


def report(repeats=16):
    r=report_fixture();calls=6*(repeats+6)
    r.update(schema="tc-qwen-frontier-fixture-v2",natural_two_block_forward=False,next_norm="reused-layer0-modulation-fixture",current_block_qkv="gpu",
        source_unchanged=True,all_outputs_bits_exact=True,single_bank=True,extra_owned_weight_bytes=0,repeats=repeats,
        timing_checksum_policy="post-loop-last-per-arm",last_owned_output_bytes=201326592)
    r["metrics"]["runtime_calls_session_total"]=calls;r["metrics"]["runtime_weight"]["hybrid_blocks_session_total"]=calls
    r["qkv"]=dict(predictions=calls,weight_stages=calls,completed_blocks=calls,validation_calls=1,validation_passed=True,
        failed=False,fallback_blocks=0,program_key="a"*64,slot_bytes=65814528,estimated_bytes=301989888)
    r["prefetch"]=dict.fromkeys(("attempts","submissions","claims","weight_completions","completed_hits"),calls)
    r["prefetch"].update(producer_submissions=3*calls,discards=0,retired_claims=0,failures=0,requested=True,enabled=True,future_pending=False)
    r["samples"]=[dict(arm=screen.ARMS[(j+rep+6)%6],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.03,
        **dict.fromkeys(screen.SPANS,.001)) for rep in range(-2,repeats) for j in range(6)]
    return r


def arrays():
    v={n:np.array([i+1.,i+2.],dtype=np.float32) for i,n in enumerate(("gpu_a","gpu_b","ffn_input_a","ffn_input_b"))}
    for a in screen.ARMS:
        for var in ("a","b","w"):
            v[f"{a}_{var}_block"]=v["gpu_"+("b" if var=="b" else "a")].copy()
            v[f"{a}_{var}_qkv"]=np.full(2,dict(a=5.,b=6.,w=7.)[var],dtype=np.float32)
    return v


class FrontierTests(unittest.TestCase):
    def test_both_bounded_sizes_keep_all_actual_predictions(self):
        for n in (16,32):
            a=screen.validate(report(n),n);self.assertEqual(a["native_calls_per_operator"],6*(n+6));self.assertEqual(a["measured_consumers"],6*n)
        for change in (lambda r:r["qkv"].update(predictions=1),lambda r:r["prefetch"].update(completed_hits=1),
            lambda r:r["metrics"]["runtime_weight"].update(fallback_blocks_session_total=1),lambda r:r.update(extra_owned_weight_bytes=1)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16)

    def test_fixture_cannot_be_relabelled_as_model_or_full40(self):
        for change in (dict(natural_two_block_forward=True),dict(complete_request=True),dict(next_norm="actual-layer1"),
            dict(completed_full_dit_steps=40),dict(timing_checksum_policy="between-samples"),dict(last_owned_output_bytes=0)):
            with self.assertRaises(ValueError):screen.validate({**report(),**change},16)

    def test_sample_order_warmup_and_complete_positive_spans(self):
        for change in (lambda r:r["samples"].reverse(),lambda r:r["samples"].pop(),lambda r:r["samples"][0].update(warmup=False),
            lambda r:r["samples"][0].update(seconds=float("nan")),lambda r:r["samples"][0].update(qkv_weight_wait_seconds=-1)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16)

    def test_two_actual_model_calls_per_scope_not_combined_credit(self):
        raw,_=fixture();template=next(x for x in raw["events"] if x["name"]=="ane.evaluate_api")
        raw["events"]=[x for x in raw["events"] if x["request"]==-2]+[dict(template,request=n,model=m) for n in screen.scopes(16) for m in (1,2)]
        raw["counts"]=dict(Counter(x["name"] for x in raw["events"]));screen.observer_contract(raw,16)
        for change in (lambda x:x["events"].pop(),lambda x:x["events"][-1].update(model=1),lambda x:x["events"][-1].update(name="ane.load_api")):
            bad=copy.deepcopy(raw);change(bad);bad["counts"]=dict(Counter(x["name"] for x in bad["events"]))
            with self.assertRaises(ValueError):screen.observer_contract(bad,16)

    def test_full_arrays_activation_source_changes_and_incremental_bits(self):
        audit.array_contract(arrays())
        for change in (lambda v:v.pop("dual-pre_w_qkv"),lambda v:v["dual-pre_a_block"].__setitem__(0,9),
            lambda v:v.update(ffn_input_b=v["ffn_input_a"].copy()),lambda v:v["original-post_w_qkv"].__setitem__(0,float("nan"))):
            v=arrays();change(v)
            with self.assertRaises(ValueError):audit.array_contract(v)

    def test_same_case_replay_does_not_pool_edit_or_timeline_performance(self):
        r=dict(case="generate",arrays=audit.array_contract(arrays()))
        audit.matched([r,copy.deepcopy(r),dict(case="edit",arrays={"full_array_sha256":{"other":"different"}})])
        bad=copy.deepcopy(r);bad["arrays"]["full_array_sha256"]["dual-post_b_qkv"]="different"
        with self.assertRaises(ValueError):audit.matched([r,bad])


if __name__=="__main__":unittest.main()
