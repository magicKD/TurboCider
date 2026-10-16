"""Offline complete real-shared FFN source/gate/order/join/cold-boundary tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"labs"))
import real_shared_ffn as shared

SPEC=importlib.util.spec_from_file_location("real_shared_curator",ROOT/"scripts/curate_real_shared_ffn.py")
curator=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(curator)


class JoinMath(unittest.TestCase):
    def test_bf16_even_ties_and_fp32_actual_component_addition(self):
        self.assertEqual(shared.bf16_rne(1+2**-8),0x3f80)
        self.assertEqual(shared.bf16_rne(1+3*2**-8),0x3f82)
        self.assertEqual(shared.bf16_rne(-(1+2**-8)),0xbf80)
        self.assertEqual(shared.cpu_join(0x3f80,0x3c00),0x4000)
        self.assertEqual(shared.cpu_join(0x3f80,0xbc00),0)
        for value in (float("nan"),float("inf"),True,1e40):
            with self.assertRaises(ValueError): shared.bf16_rne(value)
        for values in ((True,0),(0,65536),(-1,0),(0,0x7c00)):
            with self.assertRaises(ValueError): shared.cpu_join(*values)

    def test_whole_reference_header_and_source_bounds_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/"tiny.npy"
            header=repr({"descr":"<f4","fortran_order":False,"shape":(1,1)}).encode()+b"\n"
            path.write_bytes(b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+b"\0"*4)
            with self.assertRaises(ValueError): curator.reference(path)
            for offset,size in ((True,1),(0,False),(-1,1),(0,path.stat().st_size+1)):
                with self.assertRaises(ValueError): curator.span(path,offset,size)
            with self.assertRaises(ValueError): curator.local(root,"../tiny.npy")

    def test_prepared_replay_cannot_implicitly_turn_inspection_into_a_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"must-not-exist"
            args=[sys.executable,"-E","-S","-B",str(ROOT/"integrations/qwen_shared_ffn.py"),"--device","--prepared-run","missing",
                  "--fixture","missing","--checkpoint","missing","--capture-root","missing","--prior-run","missing",
                  "--point","layer-0-step-19","--output-dir",str(output)]
            child=subprocess.run(args,text=True,capture_output=True)
            self.assertEqual(child.returncode,2); self.assertIn("explicit --mode benchmark",child.stderr)
            self.assertFalse(output.exists())


class RealSharedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/"review/qwen-shared-ffn-evidence-20261016.json").read_text())

    def record(self):
        return copy.deepcopy(self.data["runs"][-1]["report"])

    def test_all_560_consumers_264_exact_joins_and_non_execution_records_retained(self):
        self.assertEqual(len(self.data["runs"]),4)
        self.assertEqual(sum(len(r["report"]["native"]["samples"]) for r in self.data["runs"]),560)
        self.assertEqual(self.data["contract_sha256"],hashlib.sha256((ROOT/"labs/real_shared_ffn.py").read_bytes()).hexdigest())
        self.assertEqual(self.data["curator_sha256"],hashlib.sha256((ROOT/"scripts/curate_real_shared_ffn.py").read_bytes()).hexdigest())
        count,stale=0,0
        for row in self.data["runs"]:
            proof=shared.verify(row["report"]); self.assertEqual(proof,row["independent_validation"])
            count+=proof["exact_runtime_join_consumers"]; stale+=proof["stale_actual_join_observations"]
            self.assertTrue(row["child_exit_confirmed"])
            self.assertTrue(row["original_spans_independently_verified"])
            for name,relative in (("qwen_witness.py","labs/ane/qwen_witness.py"),("real_bf16_source.py","labs/metal/real_bf16_source.py")):
                self.assertEqual(row["report"]["source_sha256"][name],hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())
        self.assertEqual((count,stale),(264,64))
        self.assertEqual(len(self.data["inspections"]),1); self.assertEqual(len(self.data["admission_rejections"]),1)
        self.assertEqual(curator.inspect(self.data["inspections"][0]["report"])["executed_ffn_calls"],0)
        self.assertIsNone(self.data["admission_rejections"][0]["report"]["native"])

    def test_actual_port_is_single_channel_image_not_channel_sequence_toy(self):
        value=self.record()["native"]["input_descriptor"]
        self.assertEqual([value[key] for key in ("Batches","Channels","Height","Width")],[1,1,1024,4096])
        for key,item in (("Channels",4096),("Height",1),("RowStride",16384),("Type","Float32")):
            changed=dict(value); changed[key]=item
            with self.subTest(key=key),self.assertRaises(ValueError): shared.port(changed)

    def test_exact_gpu_denominator_fixed_quality_and_full_prefix_are_protected(self):
        for kind in ("gpu","prefix","fp64","budget","dtype","layer","compile"):
            value=self.record(); native=value["native"]; sample=native["samples"][0]
            if kind=="gpu": sample["metrics"]["full_original"]["max_abs"]=.01
            elif kind=="prefix": next(s for s in native["samples"] if s["arm"]!="gpu")["metrics"]["prefix_original"]["relative_l2"]=.031
            elif kind=="fp64": sample["metrics"]["independent_full_sample"]["cosine"]=.99
            elif kind=="budget": native["gates"]["relative_l2_max"]=.1
            elif kind=="dtype": sample["output_dtype"]="float32"
            elif kind=="layer": value["source_layer"]=False
            else: native["gpu_compile_enabled"]=False
            with self.subTest(kind=kind),self.assertRaises(ValueError): shared.verify(value)

    def test_missing_order_point_cycle_or_call_cannot_hide_stale_or_reduced_work(self):
        for kind in ("call","order","point","cycle","shape"):
            value=self.record(); native=value["native"]
            if kind=="call": native["samples"].pop()
            elif kind=="order": native["samples"][:5]=native["samples"][:5][::-1]
            elif kind=="point": native["samples"][0]["point"]="layer-0-step-1"
            elif kind=="cycle": native["point_cycle"]=["layer-0-step-19"]
            else: native["shape_mhf"][0]=512
            with self.subTest(kind=kind),self.assertRaises(ValueError): shared.verify(value)

    def test_identity_copy_bytes_and_normal_owner_cleanup_are_required(self):
        for kind in ("alias","bytes","count","owner","pointer","reuse","source"):
            value=self.record(); native=value["native"]; sample=next(s for s in native["samples"] if s["arm"]=="shared-hybrid")
            if kind=="alias": sample["identity_after"]["metal_output_matches"]=False
            elif kind=="bytes": sample["external_feature_copy_bytes"]=16777216
            elif kind=="count": sample["external_feature_copy_count"]=2
            elif kind=="owner": native["restore_adoption_releases"]=1
            elif kind=="pointer": native["restore_deleter_pointer_matches"]=False
            elif kind=="reuse": native["mapping_reuse_after_failure"]=True
            else: value["original_spans_unchanged"]=False
            with self.subTest(kind=kind),self.assertRaises(ValueError): shared.verify(value)

    def test_complete_host_dependency_does_not_grant_physical_overlap(self):
        value=self.record(); proof=shared.verify(value)
        self.assertTrue(any(row["api_host_intersection_ns"]>0 for row in proof["host_intersections"] if row["arm"].endswith("hybrid")))
        self.assertTrue(all(row["api_host_intersection_ns"]==0 for row in proof["host_intersections"] if row["arm"].endswith("serial")))
        self.assertFalse(proof["physical_overlap_qualified"])
        for kind in ("clock","duration","join","physical"):
            value=self.record(); sample=next(s for s in value["native"]["samples"] if s["arm"]=="shared-hybrid")
            if kind=="clock": sample["host_events"][0]["clock_domain"]="gpu-clock"
            elif kind=="duration": sample["host_events"][0]["duration_ns"]=True
            elif kind=="join": next(e for e in sample["host_events"] if "restore" in e["name"])["start_ns"]=0
            else: sample["physical_overlap_qualified"]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError): shared.verify(value)

    def test_actual_runtime_join_and_previous_actual_output_control_are_strict(self):
        for kind in ("mismatch","hash","elements","stale","dispatch"):
            value=self.record(); sample=next(s for s in value["native"]["samples"] if "stale_actual_join_control" in s)
            if kind=="mismatch": sample["exact_runtime_join"]["exact_mismatches"]=1
            elif kind=="hash": sample["exact_runtime_join"]["actual_bf16_sha256"]="0"*64
            elif kind=="elements": sample["exact_runtime_join"]["elements"]=1
            elif kind=="stale": sample["stale_actual_join_control"]["mismatched_elements"]=0
            else: sample["stale_actual_join_control"]["new_device_dispatch_performed"]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError): shared.verify(value)
        earlier=copy.deepcopy(self.data["runs"][0]["report"]); earlier["native"]["full_exact_runtime_join_witness"]=True
        with self.assertRaises(ValueError): shared.verify(earlier)

    def test_load_selector_without_compile_is_not_zero_setup_cost(self):
        late=self.data["runs"][2:]
        self.assertTrue(all(row["report"]["native"]["compile_selector_called"] is False for row in late))
        self.assertTrue(all(row["report"]["native"]["compile_wall_ns"] is None for row in late))
        loads=[row["report"]["native"]["load_wall_ns"] for row in late]
        self.assertGreater(loads[0],10*loads[1])
        value=self.record(); value["native"]["compile_wall_ns"]=0
        with self.assertRaises(ValueError): shared.verify(value)

    def test_changed_ratios_and_physical_or_lossless_relabelling_are_refused(self):
        value=self.record(); value["comparisons"][3]["median_paired_host_ratio"]+=.1
        with self.assertRaises(ValueError): shared.verify(value)
        value=self.record(); value["production_qualified"]=True
        with self.assertRaises(ValueError): shared.verify(value)
        # Early copy/shared direction differs; both observations stay in the record.
        self.assertLess(self.data["runs"][0]["report"]["comparisons"][4]["median_paired_host_ratio"],1)
        self.assertGreater(self.data["runs"][-1]["report"]["comparisons"][4]["median_paired_host_ratio"],1)


if __name__=="__main__": unittest.main()
