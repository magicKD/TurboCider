from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_spatial_decode_screen import validate_causal_bank,validate_causal_observation,CHECKPOINT_SHA


class CausalConvolutionGuards(unittest.TestCase):
    def bank(self):
        return dict(schema="turbocider-seedvr2-vae-conv-probe-v1",checkpoint_sha256=CHECKPOINT_SHA,bias_applied=False,
            precision="fp16",source_operand_recipe="original-fp16-temporal-channel-stack",original_shape=[128,256,3,3,3],
            input_layout="N(T*C)HW",output_layout="NOHW",temporal_taps=3,input_channels=256,output_channels=128,
            total_output_channels=128,rows=64,cols=64,input_rows=66,input_cols=66,layer="decoder.up_blocks.3.resnets.0.conv1",
            qualified=False,compute_units="cpuAndNeuralEngine")
    def observation(self):
        return dict(causal_conv_backend="ne",causal_conv_frames=24,causal_conv_coreml_calls=6144,
            causal_conv_backing_hits=6144,causal_conv_slot_bytes=21626880,qualified=False,physical_ane_occupancy_measured=False)
    def test_initial_frame_is_not_relabelled_and_all_outputs_required(self):
        validate_causal_bank(self.bank());validate_causal_observation(self.observation(),"ne",self.bank(),25,1024,1024)
        for change in ({"output_channels":64},{"qualified":True},{"bias_applied":True},{"layer":"other"},{"rows":True}):
            with self.assertRaises(ValueError):validate_causal_bank({**self.bank(),**change})
        for change in ({"causal_conv_frames":25},{"causal_conv_coreml_calls":256},{"causal_conv_backing_hits":6143},
            {"causal_conv_backend":"cpu"},{"causal_conv_slot_bytes":0},{"physical_ane_occupancy_measured":True}):
            with self.assertRaises(ValueError):validate_causal_observation({**self.observation(),**change},"ne",self.bank(),25,1024,1024)
    def test_ragged_tiles_and_cpu_control_use_real_counts(self):
        observation={**self.observation(),"causal_conv_backend":"cpu","causal_conv_frames":8,
            "causal_conv_coreml_calls":32,"causal_conv_backing_hits":32}
        validate_causal_observation(observation,"cpu",self.bank(),9,65,71)
    def test_no_surface_cpu_pointer_access_in_the_actual_executor(self):
        code=(ROOT/"tools/native/seedvr2_vae_ane_causal_conv.mm").read_text()
        self.assertNotIn(".dataPointer",code)
        self.assertIn("out.pixelBuffer==slot.output.pixel",code)
        self.assertIn("tc_seedvr2_ane_causal_pack",(ROOT/"tools/native/check_release_binary.py").read_text())


if __name__=="__main__":unittest.main()
