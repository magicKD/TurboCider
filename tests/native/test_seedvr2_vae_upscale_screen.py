#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
from seedvr2_vae_upscale_screen import validate_manifest,validate_observation,SCHEMA,LAYER,CHECKPOINT_SHA

def manifest():
    return dict(schema=SCHEMA,qualified=False,checkpoint_sha256=CHECKPOINT_SHA,layer=LAYER,
        original_shape=[1024,256,1,1,1],precision="fp16",bias_applied=False,
        channel_order="spatial_row_spatial_col_channel",compute_units="cpuAndNeuralEngine",
        variant="projection",rows=64,cols=64,input_shape=[1,256,64,64],output_shape=[1,1024,64,64])

def observation():
    quality=dict(passed=True,finite=True,relative_l2=.001,cosine=.999999)
    return dict(schema="seedvr2-upscale-boundary-observation-v1",route="ne_pipeline",rows=128,cols=128,tiles=4,
        variant="projection",physical_ane_occupancy_measured=False,signed_input_aba_checked=True,quality=quality,
        coreml_calls=24,output_backing_identity_hits=24,slot_bytes=20<<20,
        bitwise_pack_checked_values=128*128*256,bitwise_consumer_checked_values=128*128*1024,
        observations=[dict(warm=i>=3,sign=-1 if i%3==1 else 1,total_seconds=.01,quality=deepcopy(quality)) for i in range(6)])

class ScreenTests(unittest.TestCase):
    def test_original_full_bank_and_shuffle_shape(self):
        value=manifest();validate_manifest(value)
        validate_manifest({**value,"variant":"shuffle","output_shape":[1,256,128,128]})
        for change in ({"precision":"bf16"},{"bias_applied":True},{"channel_order":"channel_phase"},
                       {"qualified":True},{"rows":True},{"output_shape":[1,64,64,64]}):
            with self.assertRaises(ValueError):validate_manifest({**value,**change})

    def test_complete_calls_layout_and_warm_signed_input_coverage(self):
        value=observation();validate_observation(value,"ne_pipeline",manifest(),128,3)
        for change in ({"coreml_calls":23},{"output_backing_identity_hits":0},
                       {"bitwise_consumer_checked_values":1},{"physical_ane_occupancy_measured":True}):
            with self.assertRaises(ValueError):validate_observation({**value,**change},"ne_pipeline",manifest(),128,3)
        value["observations"][1]["sign"]=1
        with self.assertRaises(ValueError):validate_observation(value,"ne_pipeline",manifest(),128,3)

    def test_numerical_gate_rejects_one_bad_or_nonfinite_request(self):
        for change in ({"relative_l2":.05001},{"cosine":.99899},{"cosine":float("nan")}):
            value=observation();value["observations"][-1]["quality"].update(change)
            with self.assertRaises(ValueError):validate_observation(value,"ne_pipeline",manifest(),128,3)

    def test_fused_gpu_requires_complete_native_word_witness(self):
        value={**observation(),"route":"gpu_fused","coreml_calls":0,"output_backing_identity_hits":0,
            "slot_bytes":0,"bitwise_default_checked_values":3*128*128*1024}
        validate_observation(value,"gpu_fused",manifest(),128,3)
        value["bitwise_default_checked_values"]-=1
        with self.assertRaises(ValueError):validate_observation(value,"gpu_fused",manifest(),128,3)

if __name__=="__main__":unittest.main()
