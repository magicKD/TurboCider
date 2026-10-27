"""Host-only complete consumer/source/work gates for QKV+Private composition."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_qkv_private_combo_screen as screen
import qwen_qkv_private_combo_audit as audit


class ComboScreenTests(unittest.TestCase):
    def report(self):
        def q(exact):return dict(relative_l2=0. if exact else .0006,cosine=1.,max_abs=0. if exact else .01,
                                 changed=0 if exact else 100,elements=1024*4096)
        r=dict(status="measured",runtime_lora_applied_terms=227,image_export=False,borrowed_values_unchanged=True,
            fresh_gpu_arguments_verified=True,private_fresh_input_restoration=True,preflight_private_calls=3,successful_private_calls=39,
            static_prefix_channels=2048,shared_ranks=False,completed_full_dit_steps=1,stopped_step=1,stopped_layer=0,
            qkv_packed_bytes=100663296,samples=[])
        for i in range(-3,15):
            for visit in range(5):
                name=screen.NAMES[(visit+i+5)%5];ffn=name in ("gpu-staged","private","private-qkv")
                r["samples"].append(dict(iteration=i,arm=name,seconds={"gpu-original":.035,"gpu-staged":.036,"gpu-qkv":.034,
                    "private":.033,"private-qkv":.032}[name],quality=q(name in screen.NAMES[:2]),
                    matched_input_ffn_quality=q(name=="gpu-staged") if ffn else None,
                    original_chain_ffn_quality=q(name=="gpu-staged") if ffn else None))
        return r

    def fenced_report(self):
        r=self.report();r.update(schedule_recipe="producer-fence-v1",input_ready_fences=34,measured_samples_per_arm=14,successful_private_calls=71)
        def q(exact):return dict(relative_l2=0. if exact else .0006,cosine=1.,max_abs=0. if exact else .01,
                                 changed=0 if exact else 100,elements=1024*4096)
        r["samples"]=[]
        for i in range(-3,14):
            for visit in range(7):
                name=screen.FENCE_NAMES[(visit+i+7)%7];private=name.startswith("private");ffn=name=="gpu-staged" or private
                r["samples"].append(dict(iteration=i,arm=name,seconds=.035 if name=="gpu-original" else .032,
                    quality=q(name in screen.NAMES[:2]),matched_input_ffn_quality=q(name=="gpu-staged") if ffn else None,
                    original_chain_ffn_quality=q(name=="gpu-staged") if ffn else None,producer_seconds=.012 if ffn else None,
                    private_phases=dict(corrections_ready_seconds=.001,submission_seconds=.0005,input_ready_wait_seconds=.0002 if name.endswith("fenced") else 0,
                        exposed_wait_seconds=.016,post_join_seconds=.002,producer_fence=name.endswith("fenced")) if private else None))
        return r

    def test_producer_fence_waits_are_actual_work_and_keep_legacy_recipe(self):
        result=screen.analyze(self.fenced_report());self.assertEqual(result["primary_recipe"],"private-qkv-fenced")
        self.assertTrue(all(len(v)==14 for v in result["whole_block_seconds"].values()))
        self.assertIn("private_phase_medians",result)
        for key,value in (("input_ready_fences",33),("successful_private_calls",39),("measured_samples_per_arm",15)):
            r=self.fenced_report();r[key]=value
            with self.assertRaises(ValueError):screen.analyze(r)
        r=self.fenced_report();next(x for x in r["samples"] if x["arm"]=="private-qkv-fenced")["private_phases"]["input_ready_wait_seconds"]=0
        with self.assertRaises(ValueError):screen.analyze(r)

    def event_report(self):
        r=self.fenced_report();r.update(schedule_recipe="producer-event-v2",input_ready_fences=37,event_value_fences=37,
            preflight_ready_wait="event-value",ready_before_private_completion=37)
        for row in r["samples"]:
            if row["arm"] in screen.FENCE_NAMES[-2:]:row["arm"]=screen.EVENT_NAMES[screen.FENCE_NAMES.index(row["arm"])]
            if row["private_phases"] is not None:
                fenced=row["arm"] in screen.EVENT_NAMES[-2:]
                row["private_phases"].update(ready_wait_kind="event-value" if fenced else "none",private_complete_at_ready=False if fenced else None)
        return r

    def test_event_value_recipe_keeps_preflight_and_early_ready_observations(self):
        result=screen.analyze(self.event_report());self.assertEqual(result["primary_recipe"],"private-qkv-event")
        self.assertEqual(result["event_ready_before_private_completion_warm_measured"],34)
        for key,value in (("input_ready_fences",34),("event_value_fences",0),("preflight_ready_wait","queue-tail"),
                          ("ready_before_private_completion",0),("ready_before_private_completion",33)):
            r=self.event_report();r[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.analyze(r)

    def test_event_ready_kind_or_physical_observation_cannot_be_fabricated(self):
        for key,value in (("ready_wait_kind","queue-tail"),("private_complete_at_ready",None),("private_complete_at_ready",0)):
            r=self.event_report();next(x for x in r["samples"] if x["arm"]=="private-qkv-event")["private_phases"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.analyze(r)
        r=self.event_report();next(x for x in r["samples"] if x["arm"]=="private")["private_phases"]["private_complete_at_ready"]=False
        with self.assertRaises(ValueError):screen.analyze(r)

    def ordered_report(self):
        r=self.event_report();templates={row["arm"]:row for row in r["samples"]}
        r.update(schedule_recipe="submission-order-v3",input_ready_fences=45,event_value_fences=45,successful_private_calls=132,
            ready_before_private_completion=45,measured_samples_per_arm=18,preflight_private_calls=6,
            gpu_first_submissions=45,gpu_first_fresh_input_restoration=True,preflight_ready_wait="event-value-and-gpu-first")
        r["samples"]=[]
        for i in range(-3,18):
            for visit in range(9):
                name=screen.ORDER_NAMES[(visit+i+9)%9];source=name.removesuffix("-gpu-first")
                row=copy.deepcopy(templates[source]);row.update(iteration=i,arm=name)
                if row["private_phases"] is not None:
                    row["private_phases"].update(submission_order="gpu-first" if name.endswith("gpu-first") else "transfer-first",
                        gpu_submit_seconds=.0001,transfer_submit_seconds=.0002)
                r["samples"].append(row)
        return r

    def test_submission_order_keeps_controls_and_uses_full_gpu_denominator(self):
        result=screen.analyze(self.ordered_report());self.assertEqual(result["primary_recipe"],"private-qkv-gpu-first")
        self.assertTrue(all(len(v)==18 for v in result["whole_block_seconds"].values()))
        self.assertEqual(result["event_ready_before_private_completion_warm_measured"],42)
        self.assertAlmostEqual(result["reduction_vs_original_gpu_percent"]["private-qkv-gpu-first"],100*(1-.032/.035))
        for key,value in (("gpu_first_submissions",42),("preflight_private_calls",3),("successful_private_calls",126),
                          ("gpu_first_fresh_input_restoration",False),("event_value_fences",42)):
            r=self.ordered_report();r[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.analyze(r)

    def test_gpu_first_order_and_both_actual_submission_spans_are_checked(self):
        for key,value in (("submission_order","transfer-first"),("gpu_submit_seconds",0),("transfer_submit_seconds",float("nan"))):
            r=self.ordered_report();next(x for x in r["samples"] if x["arm"]=="private-qkv-gpu-first")["private_phases"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.analyze(r)

    def packed_report(self):
        r=self.ordered_report();r.update(schedule_recipe="packed-suffix-v4",preflight_ready_wait="event-value-and-packed-suffix",
            packed_suffix_calls=45,packed_suffix_bytes=167772160,packed_suffix_control_exact=True,
            packed_suffix_fresh_input_restoration=True,packed_suffix_setup_seconds=.6,gpu_first_submissions=0)
        for row in r["samples"]:
            if row["arm"] in screen.ORDER_NAMES[-2:]:row["arm"]=screen.PACK_NAMES[screen.ORDER_NAMES.index(row["arm"])]
            if row["private_phases"] is not None:
                row["private_phases"].update(submission_order="transfer-first",gpu_suffix_recipe="packed-base" if row["arm"].endswith("packed") else "original")
        return r

    def test_packed_suffix_preserves_exact_controls_and_owning_actual_calls(self):
        result=screen.analyze(self.packed_report());self.assertEqual(result["primary_recipe"],"private-qkv-packed")
        for key,value in (("packed_suffix_calls",42),("packed_suffix_bytes",0),("packed_suffix_control_exact",False),
                          ("packed_suffix_fresh_input_restoration",False),("gpu_first_submissions",45),("packed_suffix_setup_seconds",float("nan"))):
            r=self.packed_report();r[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.analyze(r)

    def test_packed_suffix_recipe_is_not_a_gpu_first_or_original_suffix_label(self):
        for key,value in (("gpu_suffix_recipe","original"),("submission_order","gpu-first")):
            r=self.packed_report();next(x for x in r["samples"] if x["arm"]=="private-qkv-packed")["private_phases"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):screen.analyze(r)

    def test_raw_packed_control_geometry_does_not_relax_the_block_abi(self):
        for name,width in (("partial_reference",4096),("partial_packed",4096),("hidden_reference",10240),("hidden_packed",10240)):
            entry=dict(dtype="BF16",shape=[1,1024,width],data_offsets=[0,1024*width*2])
            self.assertEqual(audit.packed_control_geometry(name,entry)[0],[1,1024,width])
            for key,value in (("dtype","F16"),("shape",[1,1024,width//2]),("data_offsets",[0,1024*width*2-1])):
                wrong={**entry,key:value}
                with self.subTest(name=name,key=key),self.assertRaises(ValueError):audit.packed_control_geometry(name,wrong)
        with self.assertRaises(ValueError):audit.packed_control_geometry("input",entry)

    def test_combination_uses_original_gpu_denominator_and_separate_increment(self):
        r=screen.analyze(self.report())
        self.assertAlmostEqual(r["reduction_vs_original_gpu_percent"]["private-qkv"],100*(1-.032/.035))
        self.assertAlmostEqual(r["qkv_incremental_reduction_vs_private_percent"],100*(1-.032/.033))
        self.assertTrue(all(len(v)==15 for v in r["whole_block_seconds"].values()))

    def test_actual_jobs_and_ownership_preflight_cannot_be_omitted(self):
        for key,value in (("successful_private_calls",36),("preflight_private_calls",0),("private_fresh_input_restoration",False),
                          ("fresh_gpu_arguments_verified",False),("runtime_lora_applied_terms",226),("image_export",True),
                          ("borrowed_values_unchanged",False),("static_prefix_channels",1024),("shared_ranks",True)):
            r=self.report();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):screen.analyze(r)

    def test_full_block_and_matched_chain_ffn_gates_all_apply(self):
        for field,limit in (("quality",.001),("matched_input_ffn_quality",.03),("original_chain_ffn_quality",.03)):
            r=self.report();row=next(x for x in r["samples"] if x["arm"]=="private-qkv")
            row[field]["relative_l2"]=limit+.0001
            with self.assertRaises(ValueError):screen.analyze(r)
        r=self.report();r["samples"][3]["quality"]["changed"]=1
        with self.assertRaises(ValueError):screen.analyze(r)

    def test_missing_unbalanced_or_nonfinite_samples_fail(self):
        r=self.report();r["samples"].pop()
        with self.assertRaises(ValueError):screen.analyze(r)
        for value in (True,0,-1,float("nan"),float("inf")):
            r=self.report();r["samples"][0]["seconds"]=value
            with self.assertRaises(ValueError):screen.analyze(r)


if __name__=="__main__":unittest.main()
