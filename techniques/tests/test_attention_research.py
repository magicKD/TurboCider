"""Attention math/source/contract tests; no GPU or model fixtures required."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


reference=module("attention_reference","labs/metal/attention_math.py")
source=module("attention_source_contract","labs/metal/attention_source.py")
curator=module("attention_curator_contract","scripts/curate_attention_campaign.py")


class AttentionMathTests(unittest.TestCase):
    def test_dense_stable_row_and_explicit_empty_convention(self):
        result=reference.row_oracle([1.,0.],[[1.,0.],[0.,1.]],[[1.,3.],[2.,4.]],[True,True],1.)
        p=math.e/(math.e+1.)
        self.assertAlmostEqual(result[0],p+2*(1-p))
        self.assertAlmostEqual(result[1],3*p+4*(1-p))
        self.assertEqual(reference.row_oracle([10000.],[[1.],[2.]],[[3.],[5.]],[True,True],1.),[5.])
        self.assertEqual(reference.row_oracle([1.],[[1.]],[[7.,8.]],[False],1.),[0.,0.])

    def test_causal_uses_bottom_right_offset_not_top_left(self):
        self.assertEqual(reference.allowed_keys(0,3,5,"causal"),[True,True,True,False,False])
        self.assertEqual(reference.allowed_keys(2,3,5,"causal"),[True]*5)

    def test_qwen_segments_preserve_full_target_and_causal_prefix(self):
        segments=[{"start":0,"end":2,"causal":True},{"start":2,"end":5,"causal":False}]
        self.assertEqual(reference.allowed_keys(0,5,5,"segments",segments),[True,False,False,False,False])
        self.assertEqual(reference.allowed_keys(2,5,5,"segments",segments),[True]*5)
        for change in ([{"start":1,"end":5,"causal":False}],segments[:1],
                       [{"start":0,"end":5,"causal":1}]):
            with self.assertRaises(ValueError):reference.validate_segments(change,5)

    def test_row_bad_geometry_mask_types_and_nonfinite_scores_rejected(self):
        for args in (([1.],[[1.,2.]],[[1.]],[True],1.),([1.],[[1.]],[[1.]],[1],1.),
                     ([float("inf")],[[1.]],[[1.]],[True],1.)):
            with self.assertRaises(ValueError):reference.row_oracle(*args)

    def test_source_paths_and_duplicate_header_keys_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"data").write_bytes(b"fixture")
            self.assertEqual(source.local(root,"data"),(root/"data").resolve())
            for name in ("../data",str(root/"data")):
                with self.assertRaises(ValueError):source.local(root,name)
        with self.assertRaises(ValueError):source.unique_object([("q",1),("q",2)])


class AttentionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=json.loads((ROOT/"review/qwen-attention-evidence-20261016.json").read_text())

    def real(self):
        return copy.deepcopy(next(run for run in self.corpus["runs"] if run["group"]=="discovery"))

    def test_corpus_preserves_all_comparisons_rejections_and_immutable_oracle(self):
        data=self.corpus
        self.assertEqual(data["curator_sha256"],hashlib.sha256((ROOT/"scripts/curate_attention_campaign.py").read_bytes()).hexdigest())
        self.assertEqual(len(data["captures"]),2)
        self.assertEqual(len(data["runs"]),21)
        self.assertEqual(sum(run["status"]=="component-numeric-rejected" for run in data["runs"]),2)
        oracle_sha=hashlib.sha256((ROOT/"labs/metal/attention_math.py").read_bytes()).hexdigest()
        for run in data["runs"]:
            expected=run["independent_statistics"]
            declared=None if expected is None else {key:expected[key] for key in ("paired_ratios","median_paired_host_ratio")}
            self.assertEqual(curator.verify_statistics({**run,"statistics":declared}),expected)
            self.assertFalse(run["production_qualified"])
            snapshot=next(item for item in run["source_snapshots"] if item["path"].endswith("attention_math.py"))
            self.assertEqual(snapshot["sha256"],oracle_sha)

    def test_failed_gate_marker_or_nonpositive_time_cannot_pass(self):
        for change in ("time","gate","precision"):
            data=self.real()
            if change=="time":data["samples"][0]["wall_ns"]=True
            elif change=="gate":data["samples"][0]["independent_rows"]["relative_l2"]=.1
            else:data["gates"]["full_original_relative_l2_max"]=1.
            with self.assertRaises(ValueError):curator.verify_statistics(data)

    def test_failed_semantics_cannot_expose_speed_ratio(self):
        data=copy.deepcopy(next(run for run in self.corpus["runs"] if run["status"]=="component-numeric-rejected"))
        self.assertIsNone(data["independent_statistics"])
        data["statistics"]={"median_paired_host_ratio":2.}
        with self.assertRaises(ValueError):curator.verify_statistics(data)

    def test_pair_point_and_order_mismatches_are_rejected(self):
        for change in ("point","order","pair"):
            data=self.real()
            if change=="point":data["samples"][5]["point"]="other-input"
            elif change=="order":data["samples"].reverse()
            else:data["samples"][5]["pair"]=99
            with self.assertRaises(ValueError):curator.verify_statistics(data)

    def test_real_shapes_layout_and_capture_generation_are_bound(self):
        data=self.real()
        capture=next(item for item in self.corpus["captures"] if item["raw_receipt_sha256"]==data["source_identity"]["receipt_sha256"])
        curator.verify_source_link(data,capture,"discovery")
        for change in ("heads","layout","seed","segments"):
            value=copy.deepcopy(data)
            if change=="heads":value["shape_q"][1]=4
            elif change=="layout":value["gpu_layouts"][0]["actual_element_strides"][-1]=2
            elif change=="seed":value["source_identity"]["source_seed"]=43
            else:value["segments"]=[]
            with self.assertRaises(ValueError):curator.verify_source_link(value,capture,"discovery")

    def test_capture_is_replay_with_whole_block_and_actual_prefix_controls(self):
        for capture in self.corpus["captures"]:
            self.assertIn("not unchanged internal kernel capture",capture["capture_kind"])
            self.assertLessEqual(capture["driver_report"]["raw_capture_bytes"],128*1024*1024)
            self.assertEqual(capture["driver_report"]["stopped_step"],1)
            self.assertFalse(capture["driver_report"]["image_export"])
            for record in capture["driver_report"]["records"]:
                self.assertEqual(record["full_block_relative_l2"],0.)
                self.assertEqual(record["ordinary_block_output_sha256"],record["replayed_block_output_sha256"])
                self.assertTrue(record["actual_prefix_exact"])

    def test_identity_and_empty_row_diagnostics_are_not_qualification(self):
        identities=[run for run in self.corpus["runs"] if run["candidate"]=="noop"]
        self.assertEqual(len(identities),2)
        self.assertTrue(all(run["independent_statistics"]["identity_control"] for run in identities))
        diagnostic=self.corpus["diagnostics"][0]["report"]
        self.assertFalse(diagnostic["formal_performance_score"])
        self.assertEqual(len(diagnostic["cases"]),4)
        self.assertTrue(all(not case["local_zero_row_gate_pass"] for case in diagnostic["cases"]))
        self.assertEqual(diagnostic["cases"][0]["masked_first_row_max"],29/32)

    def test_device_selector_routes_to_attention_code_not_an_unrelated_card(self):
        catalog=json.loads((ROOT/"labs/catalog.json").read_text())
        cards=[card for card in catalog["entries"] if "mlx:attention-replay" in card["cases"]]
        self.assertEqual([card["chapter"] for card in cards],["metal/attention.md"])
        self.assertIn("labs/metal/attention_bench.py",cards[0]["code"])
        self.assertIn("labs/metal/attention_math.py",cards[0]["code"])


if __name__=="__main__":unittest.main()
