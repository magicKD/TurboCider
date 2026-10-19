"""Host-only full async/sync workload, finite/drain and lifecycle receipts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_async_step_pair_screen as screen
import test_qwen_lora_swiglu_request_screen as fixture


class AsyncStepPairScreenTests(unittest.TestCase):
    def report(self,seq):
        r=fixture.SwigluRequestScreenTests().report(seq)
        for row,p in zip(r["runs"],seq):
            if not p["async_pair"]:continue
            m=row["metrics"];m["acceleration_selection"]+="; bounded async step pairs"
            m["plan"]["algorithm_approximations"].append("qwen21_bounded_async_step_pair_experimental")
            m["qwen_async_step_pair"]=dict(enabled=True,recipe="bounded-async-step-pair-v1",max_batch_steps=2,submitted_steps=p["steps"],checked_steps=p["steps"],
                accepted_steps=p["steps"],completion_batches=(p["steps"]+1)//2,evaluation_calls=2*p["steps"],peak_pending_steps=2,peak_logical_state_bytes=262146,
                host_ahead_observations=p["steps"]//2,final_drain_complete=True,diagnostic_submit_observer=False,per_step_finite_masks=True,
                progress_after_checked_completion=True,extra_weight_bytes=0)
        return r

    def test_full_raw_operator_matrix_and_broader_both_tier_workloads(self):
        for mode,count in (("raw",32),("quality",92),("restoration",36)):
            seq=screen.points(mode);self.assertEqual(len(seq),count);screen.validate(self.report(seq),seq)
        raw=screen.points("raw")
        self.assertEqual({(p["qkv"],p["ffn"]) for p in raw},{(q,f) for q in (False,True) for f in (False,True)})

    def test_async_flip_must_prime_but_seed_alone_keeps_bank(self):
        s=screen.Sequence();s.add();s.add();self.assertTrue(s.points[-1]["prefix_hit"])
        s.add(async_pair=True);self.assertFalse(s.points[-1]["prefix_hit"])
        s.add(async_pair=True,seed=74);self.assertTrue(s.points[-1]["prefix_hit"])
        s.add("generate",6,True,False);s.add(async_pair=True)
        self.assertTrue(s.points[-1]["conditioning_hit"]);self.assertFalse(s.points[-1]["prefix_hit"])

    def test_complete_masks_waits_and_no_unchecked_progress_are_required(self):
        seq=screen.points("timing","edit",5,True,True)
        for key,value in (("submitted_steps",4),("checked_steps",4),("accepted_steps",4),("completion_batches",2),("evaluation_calls",5),
            ("peak_pending_steps",3),("peak_pending_steps",2.),("peak_logical_state_bytes",0),("host_ahead_observations",3),
            ("final_drain_complete",False),("diagnostic_submit_observer",True),("per_step_finite_masks",False),("progress_after_checked_completion",False)):
            r=self.report(seq);r["runs"][1]["metrics"]["qwen_async_step_pair"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.validate(r,seq)

    def test_sync_control_cannot_claim_async(self):
        seq=screen.points("timing");r=self.report(seq);r["runs"][0]["metrics"]["qwen_async_step_pair"]={"enabled":True}
        with self.assertRaises(ValueError):screen.validate(r,seq)


if __name__=="__main__":unittest.main()
