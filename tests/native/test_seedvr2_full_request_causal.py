import argparse
from pathlib import Path
import sys
import tempfile
import unittest
import os
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_framework_screen import check_native_execution,sha256,configure_raw_capture


class FullRequestCausalChecks(unittest.TestCase):
    def test_raw_capture_requires_explicit_single_streaming_diagnostic(self):
        args=argparse.Namespace(capture_raw_video=True,route="native-gpu",kind="video",count=1,output=Path("owned"),
            vae_options=dict(encoder="streaming",decoder="streaming"))
        with patch.dict(os.environ,{},clear=True):
            configure_raw_capture(args)
            self.assertEqual(os.environ["TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE"],str(Path("owned").absolute()/"raw-pixels"))
            self.assertEqual(os.environ["TURBOCIDER_SEEDVR2_VIDEO_PIXEL_HASH"],"1")
            args.capture_raw_video=False
            with self.assertRaisesRegex(ValueError,"inherited"):configure_raw_capture(args)
        for name,bad in (("route","native-hybrid"),("kind","image"),("count",2),("vae_options",{})):
            wrong=argparse.Namespace(**{**vars(args),"capture_raw_video":True,name:bad})
            with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):configure_raw_capture(wrong)

    def test_experimental_route_is_distinct_from_pure_gpu_and_counts_all_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/"manifest.json";manifest.write_text("bound test bank")
            args=argparse.Namespace(route="native-gpu",kind="video",frames=25,width=1024,height=1024,
                vae_options=None,residency="component_staged",experimental_causal_manifest=manifest,experimental_causal_compute="ne")
            value=dict(actual_denoise_steps=1,conditioning_cache_hit=False,execution="gpu_vae_cpu_ne_experimental",
                numerical_recipe="seedvr2-stable-fp16-fp32-v1+experimental-vae-causal-coreml-original-fp16-v1",
                dit_projection_dtype="fp16",dit_residual_dtype="fp32",coreml_prefix_calls=0,
                vae_causal_conv_experimental_enabled=True,vae_causal_conv_calls=6144,vae_causal_conv_frames=24,
                vae_causal_conv_backing_hits=6144,vae_causal_conv_compute_units="cpuAndNeuralEngine",
                vae_causal_conv_manifest_sha256=sha256(manifest),physical_ane_occupancy_measured=False)
            check_native_execution(args,value)
            for change in ({"execution":"gpu"},{"vae_causal_conv_calls":256},{"vae_causal_conv_frames":25},
                {"vae_causal_conv_backing_hits":0},{"vae_causal_conv_manifest_sha256":"other"},{"physical_ane_occupancy_measured":True}):
                with self.subTest(change=change),self.assertRaises(ValueError):check_native_execution(args,{**value,**change})
            args.experimental_causal_manifest=None
            with self.assertRaises(ValueError):check_native_execution(args,value)

    def test_ordinary_build_rejects_the_selector_and_build_flag_is_identity_bound(self):
        source=(ROOT/"native/platform/apple/seedvr2_session.mm").read_text()
        self.assertIn("VAE causal convolution requires an isolated experimental build",source)
        self.assertIn("VAE upscale causal convolution requires an isolated experimental build",source)
        self.assertIn("experimental-vae-causal-coreml-original-fp16-v1",source)
        build=(ROOT/"tools/native/build.sh").read_text()
        self.assertLess(build.index("COMMON+=(-DTURBOCIDER_ENABLE_VAE_CAUSAL_CONV_EXPERIMENT=1)"),build.index("BUILD_IDENTITY_FLAGS="))

    def test_upscale_only_request_requires_all_256_outputs_and_zero_image_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/"manifest.json";manifest.write_text("independent upscale bank")
            args=argparse.Namespace(route="native-gpu",kind="video",frames=25,width=1024,height=1024,vae_options=None,
                experimental_upscale_causal_manifest=manifest,experimental_upscale_causal_compute="ne")
            value=dict(actual_denoise_steps=1,conditioning_cache_hit=False,execution="gpu_vae_cpu_ne_experimental",
                numerical_recipe="seedvr2-stable-fp16-fp32-v1+experimental-vae-upscale-causal-coreml-original-fp16-v1",
                dit_projection_dtype="fp16",dit_residual_dtype="fp32",coreml_prefix_calls=0,physical_ane_occupancy_measured=False,
                vae_upscale_causal_conv_experimental_enabled=True,vae_upscale_causal_conv_calls=6144,
                vae_upscale_causal_conv_frames=24,vae_upscale_causal_conv_backing_hits=6144,vae_upscale_causal_conv_output_channels=256,
                vae_upscale_causal_conv_layer="decoder.up_blocks.2.upsamplers.0.conv",vae_upscale_causal_conv_compute_units="cpuAndNeuralEngine",
                vae_upscale_causal_conv_manifest_sha256=sha256(manifest))
            check_native_execution(args,value)
            for change in ({"vae_upscale_causal_conv_calls":256},{"vae_upscale_causal_conv_output_channels":128},
                {"vae_upscale_causal_conv_frames":25},{"execution":"gpu"},{"vae_upscale_causal_conv_backing_hits":6143}):
                with self.subTest(change=change),self.assertRaises(ValueError):check_native_execution(args,{**value,**change})
            args.kind="image";args.frames=1
            value.update(execution="gpu",numerical_recipe="seedvr2-stable-fp16-fp32-v1",vae_upscale_causal_conv_calls=0,
                vae_upscale_causal_conv_frames=0,vae_upscale_causal_conv_backing_hits=0)
            check_native_execution(args,value)


if __name__=="__main__":unittest.main()
