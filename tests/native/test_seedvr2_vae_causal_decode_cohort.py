from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_vae_causal_decode_cohort import validate_observation


class CausalCohortGuards(unittest.TestCase):
    def value(self,route="ne"):
        return dict(schema="seedvr2-vae-causal-decode-bench-v1",status="complete",route=route,frames=25,height=1024,width=1024,
            share_upscale_history=True,qualified=False,physical_ane_occupancy_measured=False,external_slot_bytes=21626880 if route!="gpu" else 0,
            iterations=[dict(iteration=i,warm=i>0,emitted_frames=25,compared_values=78643200,coreml_calls=6144 if route!="gpu" else 0,
                causal_frames=24 if route!="gpu" else 0,initial_gpu_frame_checked=True,per_frame_quality_checked=True,quality_passed=True,unequal_values=100 if route!="gpu" else 0,
                mlx_active_peak_bytes=12592974678,seconds_with_host_quality=55.,relative_l2=.001,cosine=.999999,causal_boundary_seconds=4.) for i in range(3)])
    def test_fresh_complete_warm_iterations_are_mandatory(self):
        for route in ("gpu","ne"):validate_observation(self.value(route),route,25,1024,1024,3,dict(rows=64,cols=64))
        for change in ({"initial_gpu_frame_checked":False},{"emitted_frames":24},{"coreml_calls":256},{"causal_frames":25},
            {"warm":False},{"relative_l2":.05001},{"cosine":float("nan")}):
            value=self.value();value["iterations"][-1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_observation(value,"ne",25,1024,1024,3,dict(rows=64,cols=64))
    def test_unfinished_or_gpu_unequal_outputs_are_rejected(self):
        value=self.value();value["status"]="running"
        with self.assertRaises(ValueError):validate_observation(value,"ne",25,1024,1024,3,dict(rows=64,cols=64))
        value=self.value("gpu");value["iterations"][-1]["unequal_values"]=1
        with self.assertRaises(ValueError):validate_observation(value,"gpu",25,1024,1024,3,dict(rows=64,cols=64))

    def test_dual_route_requires_both_complete_layers_and_initial_gpu_frame(self):
        value=self.value("dual");value.update(upscale_layer="decoder.up_blocks.2.upsamplers.0.conv",upscale_external_slot_bytes=24000000,external_slot_bytes=45626880)
        for row in value["iterations"]:row.update(upscale_coreml_calls=6144,upscale_causal_frames=24,total_coreml_calls=12288,upscale_causal_boundary_seconds=6.)
        bank=dict(rows=64,cols=64,layer=value["upscale_layer"])
        validate_observation(value,"dual",25,1024,1024,3,bank,bank)
        for change in ({"upscale_coreml_calls":256},{"total_coreml_calls":6144},{"upscale_causal_frames":25},
            {"initial_gpu_frame_checked":False},{"per_frame_quality_checked":False},{"upscale_causal_boundary_seconds":float("nan")}):
            changed={**value,"iterations":[dict(row) for row in value["iterations"]]};changed["iterations"][-1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate_observation(changed,"dual",25,1024,1024,3,bank,bank)
        with self.assertRaises(ValueError):validate_observation(value,"dual",25,1024,1024,3,bank)


if __name__=="__main__":unittest.main()
