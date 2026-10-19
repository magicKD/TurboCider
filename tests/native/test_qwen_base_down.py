"""CPU-only admission and sample controls for original-pitch F32 down tiles."""
import copy
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/profiling/qwen_base"))
import handoff_collect as screen
import down_audit
from test_qwen_base_handoff import report_fixture


def fixture(policy="m32n128d",channels=5120):
    r=report_fixture(channels);bm,bn,static=screen.DOWN_POLICIES[policy]
    r["gpu_down_policy"]=dict(name=policy,bm=bm,bn=bn,static_tiles=static,physical_weight_pitch=12288,
        input_width=12288-channels,output_dtype="float32",weight_copies=0)
    r.update(down_quality_a=dict(relative_l2=1e-6,elements=4194304),down_quality_b=dict(relative_l2=1e-6,elements=4194304))
    arms=("gate-up-alone","down-alone","control-down-alone")
    r["gpu_down_component_samples"]=[dict(arm=arms[(j+rep+3)%3],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.003)
        for rep in range(-2,8) for j in range(3)]
    return r


class DownTests(unittest.TestCase):
    def test_bounded_explicit_tile_matrix_and_no_silent_default(self):
        self.assertEqual(len(screen.DOWN_POLICIES),12)
        for policy in screen.DOWN_POLICIES:
            r=fixture(policy);result=screen.validate(r,8,5120,down_policy=policy)
            self.assertEqual(result["gpu_down_component_median_seconds"]["down-alone"],.003)
            with self.assertRaises(ValueError):screen.validate(r,8,5120)
        for bad in (True,{},"auto","m128n128d"):
            with self.assertRaises((ValueError,TypeError)):screen.down_analysis({},8,5120,bad)

    def test_pitch_precision_copy_and_policy_must_match_compiled_receipt(self):
        r=fixture()
        for key,value in (("physical_weight_pitch",7168),("output_dtype","bfloat16"),("weight_copies",1),
            ("input_width",6144),("name","m64n128d"),("static_tiles",0),("bm",True)):
            changed=copy.deepcopy(r);changed["gpu_down_policy"][key]=value
            with self.assertRaises(ValueError):screen.down_analysis(changed,8,5120,"m32n128d")

    def test_all_finite_rotated_intervals_and_quality_controls_are_required(self):
        r=fixture()
        changes=(lambda x:x["gpu_down_component_samples"].pop(),lambda x:x["gpu_down_component_samples"].reverse(),
            lambda x:x["gpu_down_component_samples"][0].update(seconds=float("nan")),
            lambda x:x["gpu_down_component_samples"][0].update(input_variant=True),
            lambda x:x["down_quality_a"].update(relative_l2=.001),lambda x:x["down_quality_b"].update(elements=1))
        for change in changes:
            changed=copy.deepcopy(r);change(changed)
            with self.assertRaises(ValueError):screen.down_analysis(changed,8,5120,"m32n128d")

    def test_half_gate_up_is_explicit_approximate_and_keeps_runtime_weight_arguments(self):
        r=fixture();r.update(gpu_gate_up_precision="fp16",gpu_gate_up_weight_binding="runtime-arguments-per-call-cast",
            gate_up_quality_a=dict(relative_l2=.001,elements=1024*7168),gate_up_quality_b=dict(relative_l2=.001,elements=1024*7168))
        screen.validate(r,8,5120,down_policy="m32n128d",gate_up="fp16")
        with self.assertRaises(ValueError):screen.validate(r,8,5120,down_policy="m32n128d")
        for change in (lambda x:x.update(gpu_gate_up_weight_binding="captured-half-bank"),
            lambda x:x["gate_up_quality_a"].update(relative_l2=.02),
            lambda x:x["gate_up_quality_b"].update(elements=1),
            lambda x:x["gate_up_quality_a"].update(relative_l2=float("nan"))):
            changed=copy.deepcopy(r);change(changed)
            with self.assertRaises(ValueError):screen.validate(changed,8,5120,down_policy="m32n128d",gate_up="fp16")

    def test_sampled_f64_oracle_uses_complete_k_and_original_physical_rows(self):
        dense=np.arange(21,dtype=np.float32).reshape(3,7)
        weights=(dense.view(np.uint32)>>16).astype(np.uint16)
        hidden=np.arange(15,dtype=np.float32).reshape(1,3,5)
        output=(hidden@dense[:,:5].T).astype(np.float32)
        points=[(m,n) for m in range(3) for n in range(3)]
        self.assertEqual(down_audit.sampled_dot(weights,hidden,output,points)["max_abs"],0)
        for w,h,y,p in ((weights[:,1:],hidden,output,points),(weights,hidden[:,:,:4],output,points),
            (weights,hidden,output+1,points),(weights,hidden,output,[]),
            (weights.astype(np.float32),hidden,output,points),(weights,hidden,output,[(3,0)])):
            with self.assertRaises(ValueError):down_audit.sampled_dot(w,h,y,p)

    def test_component_reader_preserves_f32_partials_and_rejects_type_or_extent_changes(self):
        hidden=np.array([1.,2.],dtype="<f4");bits=(hidden.view(np.uint32)>>16).astype("<u2")
        partial=np.array([1.000001,2.000002],dtype="<f4")
        header=dict(hidden=dict(dtype="BF16",shape=[2],data_offsets=[0,4]),down=dict(dtype="F32",shape=[2],data_offsets=[4,12]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"values.safetensors"
            def save(h):
                raw=json.dumps(h).encode();path.write_bytes(struct.pack("<Q",len(raw))+raw+bits.tobytes()+partial.tobytes())
            save(header);spec=dict(hidden=("BF16",(2,)),down=("F32",(2,)))
            values=down_audit.read_components(path,spec)
            np.testing.assert_array_equal(values["hidden"],hidden);np.testing.assert_array_equal(values["down"],partial)
            for change in (lambda h:h["down"].update(dtype="BF16"),lambda h:h["down"].update(data_offsets=[0,8]),
                lambda h:h["down"].update(shape=[1]),lambda h:h["hidden"].update(data_offsets=[1,5])):
                h=copy.deepcopy(header);change(h);save(h)
                with self.assertRaises(ValueError):down_audit.read_components(path,spec)


if __name__=="__main__":unittest.main()
