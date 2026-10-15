"""CPU contracts for packed layouts, bounded tiles and evidence statistics."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


sys.path.insert(0, str(ROOT / "labs"))
quant = module("gpu_quant_reference", "labs/quantization.py")
sys.path.pop(0)
tiles = module("gpu_affine_tiles", "labs/metal/affine_gemm.py")
sys.path.insert(0, str(ROOT / "labs/metal"))
bench = module("gpu_quant_bench", "labs/metal/quantized_bench.py")
sys.path.pop(0)
curator = module("gpu_quant_curator", "scripts/curate_gpu_campaign.py")


class FormatTests(unittest.TestCase):
    def test_q4_half_block_order_not_adjacent_nibbles(self):
        raw = struct.pack("<e", .125) + bytes(i | ((15-i)<<4) for i in range(16))
        expected = [(i-8)*.125 for i in range(16)] + [(7-i)*.125 for i in range(16)]
        self.assertEqual(quant.ggml_q4_0_block(raw), expected)
        wrong = [(q-8)*.125 for byte in raw[2:] for q in (byte&15, byte>>4)]
        self.assertNotEqual(wrong, expected)

    def test_q8_signed_extremes_and_negative_scale(self):
        codes = [-128, 127, -1, 0] * 8
        raw = struct.pack("<e", -.25) + struct.pack("<32b", *codes)
        self.assertEqual(quant.ggml_q8_0_block(raw), [-.25*x for x in codes])

    def test_wrong_byte_count_and_nonfinite_metadata_rejected(self):
        for function, size in ((quant.ggml_q4_0_block, 18), (quant.ggml_q8_0_block, 34)):
            for raw in (bytes(size-1), bytes(size+1), struct.pack("<e", float("inf"))+bytes(size-2),
                        struct.pack("<e", float("nan"))+bytes(size-2)):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    function(raw)


class TileTests(unittest.TestCase):
    def test_tail_geometry_and_padding_cost(self):
        self.assertEqual(tiles.geometry(33,41,96,4,32,32)["threads"],128)
        base = tiles.geometry(1024,24576,4096,8,64,64)
        padded = tiles.geometry(1024,24576,4096,8,64,64,8)
        self.assertEqual(padded["declared_threadgroup_bytes"] - base["declared_threadgroup_bytes"],512)
        self.assertLess(padded["declared_threadgroup_bytes"],32768)

    def test_invalid_tiles_and_group_alignment_rejected(self):
        for args in ((0,41,96,4,32,32), (33,41,95,4,32,32), (33,41,96,3,32,32),
                     (True,41,96,4,32,32), (33,41,96,4,32,128), (33,41,96,4,32,32,True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tiles.geometry(*args)

    def test_all_operand_strides_are_referenced(self):
        source = (ROOT/"labs/metal/affine_gemm.metal").read_text()
        for name in ("X_strides", "Q_strides", "S_strides", "B_strides"):
            self.assertIn(name, source)
        self.assertIn("WEIGHT_LD", source)
        self.assertIn("row < M && col < N", source)


class StatisticsTests(unittest.TestCase):
    def samples(self):
        return [{"trial":i,"warmup":False,"pair":i//2,"point":"fixture",
                 "variant":"candidate" if i%4 in (1,2) else "packed",
                 "same_recipe_numerical_pass":True,"wall_seconds":1. if i%4 in (1,2) else 2.}
                for i in range(8)]

    def test_paired_ratio_is_derived_and_deterministic(self):
        first = bench.paired(self.samples(),"packed")
        self.assertEqual(first["source_kind"],"derived")
        self.assertEqual(first["median_paired_speedup"],2.)
        self.assertEqual(first, bench.paired(self.samples(),"packed"))

    def test_invalid_or_unqualified_samples_do_not_produce_statistics(self):
        for field, invalid in (("wall_seconds",0),("wall_seconds",True),("wall_seconds",float("nan")),
                               ("same_recipe_numerical_pass",False),("point","other")):
            samples = copy.deepcopy(self.samples()); samples[1][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                bench.paired(samples,"packed")
        samples = self.samples(); samples.pop()
        with self.assertRaises(ValueError):
            bench.paired(samples,"packed")


class ExcerptTests(unittest.TestCase):
    def test_source_identity_and_license_are_retained(self):
        root = ROOT/"assets/source-excerpts/ggml-dequant-64e9bce"
        info = json.loads((root/"PROVENANCE.json").read_text())
        self.assertEqual(info["commit"],"64e9bceb2c3a856efed96feda784a50947049feb")
        self.assertEqual(info["license"],"MIT; accompanying LICENSE retained")
        self.assertIn("The ggml authors",(root/"LICENSE").read_text())
        source = (root/"dequantizers.metal").read_text()
        self.assertEqual(source.count("void dequantize_q"),2)
        self.assertNotIn("kernel void",source)


class CuratorTests(unittest.TestCase):
    def report(self):
        samples=StatisticsTests().samples()
        for sample in samples:
            sample.update(vs_same_recipe_fp32={"finite":True,"relative_l2":.001},
                          vs_independent_fp64={"finite":True,"relative_l2":.001},
                          vs_original_bf16_replica={"finite":True,"relative_l2":.05,"cosine":.995},
                          original_component_budget_pass=False)
        return {"status":"same_recipe_numeric_pass","baseline":"packed","synthetic":False,
                "repeats":8,"original_component_budget_pass":False,"samples":samples,
                "statistics":bench.paired(samples,"packed")}

    def test_same_storage_stats_do_not_inherit_original_model_quality(self):
        value=self.report()
        stats=curator.statistics_for(value)
        self.assertEqual(stats["median_paired_speedup"],2.)
        self.assertIn("same-encoded",stats["scope"])
        self.assertFalse(value["original_component_budget_pass"])

    def test_model_ratio_withheld_and_false_gate_marker_rejected(self):
        value=self.report()
        value["baseline"]="bf16"
        for sample in value["samples"]:
            if sample["variant"]=="packed":sample["variant"]="bf16"
        with self.assertRaises(ValueError):curator.statistics_for(value)
        value.pop("statistics")
        self.assertIsNone(curator.statistics_for(value))
        value=self.report(); value["samples"][0]["vs_independent_fp64"]["relative_l2"]=.2
        with self.assertRaises(ValueError):curator.statistics_for(value)


if __name__=="__main__":
    unittest.main()
