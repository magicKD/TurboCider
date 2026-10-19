"""CPU-only explicit neighboring fixed shares, provenance and matched controls."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import ffn_projection_collect as projection
import ffn_share_audit as audit
from test_qwen_base_handoff import report_fixture


class NeighborShareTests(unittest.TestCase):
    def test_actual_new_shares_require_explicit_expected_partition(self):
        for c in (5632,6144,6656):
            r=report_fixture(c);projection.validate(r,8,"descriptor","sigmoid-complement",channels=c)
            if c!=6144:
                with self.assertRaises(ValueError):projection.validate(r,8,"descriptor","sigmoid-complement")
        for c in (True,0,5888,7168,"5632"):
            with self.assertRaises(ValueError):projection.validate({},8,"descriptor",channels=c)

    def test_compiled_abi_and_native_completion_fields_cannot_be_inferred(self):
        r=report_fixture(5632);r.update(compiled_native_id="bound",neighbor_share_screen=True)
        summary=dict(channels=5632,repeats=8,projection="descriptor",activation="sigmoid-complement")
        audit.neighbor_contract(summary,r,"bound")
        for key,value in (("compiled_native_id","old"),("neighbor_share_screen",False),("handoff_screen",False),
            ("ane_channels",6144),("gpu_channels",6144),("gpu_complement_strategy","packed"),
            ("gpu_packed_weight_bytes",1024),("gpu_packed_setup_seconds",.01)):
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):audit.neighbor_contract(summary,bad,"bound")
        for field in ("prediction_wait_seconds","output_wait_seconds","caller_wait_seconds"):
            bad=copy.deepcopy(r);next(s for s in bad["samples"] if s["arm"]=="private-restaged")["completion_deltas"][field]=float("nan")
            with self.assertRaises(ValueError):audit.neighbor_contract(summary,bad,"bound")

    def test_complement_samples_are_full_and_warmup_is_never_measured(self):
        r=report_fixture(6656);r.update(compiled_native_id="bound",neighbor_share_screen=True)
        summary=dict(channels=6656,repeats=8,projection="descriptor",activation="sigmoid-complement")
        for change in (lambda x:x["gpu_complement_alone_samples"].pop(),lambda x:x["gpu_complement_alone_samples"].reverse(),
            lambda x:x["gpu_complement_alone_samples"][0].update(warmup=False),lambda x:x["gpu_complement_alone_samples"][0].update(seconds=-1)):
            bad=copy.deepcopy(r);change(bad)
            with self.assertRaises(ValueError):audit.neighbor_contract(summary,bad,"bound")

    def test_matched_source_strong_gpu_and_same_share_replays_are_distinct(self):
        records=[]
        for c,mode in ((6144,"stock"),(5632,"stock"),(6656,"stock"),(6144,"stock"),(6656,"timeline")):
            records.append(dict(cohort=str(len(records)),channels=c,case="generate",mode=mode,
                array_sha256=dict(ffn_input="same",block_reference="same",alternate_reference="same",other_weight_reference="same",block_candidate=str(c)),
                analysis=dict(median_seconds={"private-restaged":.024+c/1e7})))
        result=audit.matched(records);self.assertIsNone(result["comparisons"][-1]["reduction_vs_first6144"])
        for change in (lambda x:x.reverse(),lambda x:x[1]["array_sha256"].update(ffn_input="other"),
            lambda x:x[-1]["array_sha256"].update(block_candidate="changed"),lambda x:x[2].update(case="edit")):
            bad=copy.deepcopy(records);change(bad)
            with self.assertRaises(ValueError):audit.matched(bad)


if __name__=="__main__":unittest.main()
