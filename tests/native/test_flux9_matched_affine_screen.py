import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "tools/validation"))
from flux9_matched_affine_screen import validate_report
from flux9_matched_affine_evidence import portable_tree


class MatchedAffineScreenTests(unittest.TestCase):
    def test_portable_conversion_receipt_converts_path_keys_too(self):
        source = str(ROOT / "models/weights.safetensors")
        self.assertEqual(portable_tree(dict(source_sha256={source:"digest"})),
                         dict(source_sha256={"models/weights.safetensors":"digest"}))

    def test_native_packed_requires_all_blocks_and_never_dense_compiled_receipt(self):
        report = dict(engine="native-q4",runs=[dict(request_wall_including_export=1.0,
            actual_compiled_single_blocks=0,native_event_counts=dict(transformer_block=128),metrics=dict(
                model="flux2-klein-9b",operation="image.generate",width=512,height=512,steps=4,
                seed=42,text_tokens=512,prompt_cache_hit=index==3,hybrid={})) for index in range(4)])
        validate_report(report,"native-q4")
        for key,value in (("actual_compiled_single_blocks",96),("native_event_counts",dict(transformer_block=80))):
            bad = copy.deepcopy(report)
            bad["runs"][1][key] = value
            with self.assertRaises(ValueError):
                validate_report(bad,"native-q4")

    def test_reference_saved_bits_are_not_proof_of_actual_group_or_coverage(self):
        coverage = {key:dict(required_targets=1,covered_targets=1,missing=[]) for key in ("vae","text_encoder","transformer")}
        coverage["transformer"].update(bits=4,group_size=32,quantized_linears=153)
        report = dict(engine="mflux-q4",quantization=4,precision="mlx.core.bfloat16",required_weight_coverage=coverage,
            runs=[dict(actual_steps=[0,1,2,3],actual_quantization=4,loras=[]) for _ in range(4)])
        validate_report(report,"mflux-q4")
        for changes in (dict(group_size=64),dict(quantized_linears=152),dict(covered_targets=0),dict(missing=["weight"])):
            bad = copy.deepcopy(report)
            bad["required_weight_coverage"]["transformer"].update(changes)
            with self.assertRaises(ValueError):
                validate_report(bad,"mflux-q4")


if __name__ == "__main__":
    unittest.main()
