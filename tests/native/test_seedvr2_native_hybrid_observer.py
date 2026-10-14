#!/usr/bin/env python3
"""CPU checks for full-request route validation and no silent Core ML use."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "tools/validation"))
import seedvr2_framework_screen as observer


class NativeHybridObserverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="seedvr2-observer-")
        self.manifest = Path(self.directory.name) / "manifest.json"
        self.manifest.write_text("{}")
        self.args = SimpleNamespace(route="native-hybrid",residency="resident",kind="video",
            frames=25,fps=24,width=1024,height=1024,seed=42,ane_manifest=self.manifest)

    def tearDown(self):
        self.directory.cleanup()

    def metrics(self):
        return dict(actual_denoise_steps=1,conditioning_cache_hit=False,execution="gpu_ane",
            numerical_recipe="seedvr2-original-fp16-coreml-prefix-v1",dit_projection_dtype="fp16",dit_residual_dtype="fp32",
            coreml_prefix_calls=1792,coreml_prefix_video_blocks=32,coreml_prefix_compute_units="cpuAndNeuralEngine",
            coreml_prefix_manifest_sha256=observer.sha256(self.manifest),weight_residency_recipe="resident")

    def test_qualified_hybrid_request_and_real_count_are_bound(self):
        request = observer.native_request(self.args,Path("input.mp4"),Path("out.mp4"),2)
        self.assertEqual(request["sampling"]["seed"],44)
        self.assertEqual(request["execution"]["policy"],"gpu_ane")
        self.assertTrue(request["execution"]["allow_approximation"])
        self.assertEqual(request["execution"]["ane_manifest"],str(self.manifest))
        observer.check_native_execution(self.args,self.metrics())

    def test_missing_blocks_calls_bank_or_precision_is_rejected(self):
        for key,bad in (("coreml_prefix_calls",0),("coreml_prefix_video_blocks",1),
                        ("coreml_prefix_manifest_sha256","bad"),("numerical_recipe","seedvr2-stable-fp16-fp32-v1"),
                        ("conditioning_cache_hit",True),("execution","gpu")):
            with self.subTest(key=key):
                value = self.metrics(); value[key] = bad
                with self.assertRaises(ValueError): observer.check_native_execution(self.args,value)

    def test_pure_gpu_has_no_manifest_or_prefix(self):
        self.args.route = "native-gpu"
        request = observer.native_request(self.args,Path("input.mp4"),Path("out.mp4"),0)
        self.assertEqual(request["execution"],dict(policy="gpu",residency="resident"))
        value = self.metrics()
        value.update(execution="gpu",numerical_recipe="seedvr2-stable-fp16-fp32-v1",coreml_prefix_calls=0)
        observer.check_native_execution(self.args,value)
        value["coreml_prefix_calls"] = 1
        with self.assertRaises(ValueError): observer.check_native_execution(self.args,value)

    def test_residency_mismatch_is_rejected(self):
        self.args.residency = "component_staged"
        with self.assertRaises(ValueError): observer.check_native_execution(self.args,self.metrics())
        value = self.metrics(); value["weight_residency_recipe"] = "stage_dit"
        observer.check_native_execution(self.args,value)

    def test_request_local_vae_controls_require_matching_runtime_evidence(self):
        self.args.vae_options = dict(decoder="streaming",convolution="mixed3d",tile_rows=4096,allocator_cache_bytes=1 << 30)
        request = observer.native_request(self.args,Path("input.mp4"),Path("out.mp4"),0)
        self.assertEqual(request["execution"]["seedvr2_vae"],self.args.vae_options)
        value = self.metrics()
        value.update(vae_decoder_recipe="streaming",vae_conv_recipe="mixed3d",vae_conv_tile_rows_limit=4096,
                     allocator_cache_limit_bytes=1 << 30,streamed_output_frames=25)
        observer.check_native_execution(self.args,value)
        for key,bad in (("vae_decoder_recipe","buffered"),("vae_conv_recipe","singleton2d"),
                        ("vae_conv_tile_rows_limit",16384),("allocator_cache_limit_bytes",4 << 30),
                        ("streamed_output_frames",24)):
            wrong = dict(value); wrong[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): observer.check_native_execution(self.args,wrong)


if __name__ == "__main__": unittest.main()
