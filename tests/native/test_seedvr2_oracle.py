#!/usr/bin/env python3
"""CPU checks that the independent reference really parses its policy."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
COMFY = Path(os.environ.get("TURBOCIDER_TEST_COMFY_ROOT",ROOT/".deps/comfy-v0.39.0-20261012"))
EXTRA = Path(os.environ.get("TURBOCIDER_TEST_COMFY_EXTRA",ROOT/".deps/comfy-v0.39-runtime-extra-20261012"))
SPEC = importlib.util.spec_from_file_location("seed_oracle_fixture",ROOT/"tools/validation/seedvr2_oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


@unittest.skipUnless((COMFY/"comfy/cli_args.py").is_file() and EXTRA.is_dir(),"pinned Comfy policy fixture unavailable")
class OraclePolicyTests(unittest.TestCase):
    def configure(self,device,attention,preimport=False):
        program = r'''
import importlib.util,json,pathlib,sys
root,comfy_root,extra_root,device,attention,preimport=sys.argv[1:]
spec=importlib.util.spec_from_file_location("seed_oracle",pathlib.Path(root)/"tools/validation/seedvr2_oracle.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if preimport=="1":
    sys.path.insert(0,comfy_root)
    import comfy.cli_args
try:
    args=module.configure_comfy(pathlib.Path(comfy_root),pathlib.Path(extra_root),device,attention)
    print(json.dumps(dict(cpu=args.cpu,gpu_only=args.gpu_only,sdpa=args.use_pytorch_cross_attention)))
except ValueError as error:
    print(json.dumps(dict(error=str(error))))
'''
        value = subprocess.check_output([sys.executable,"-B","-c",program,str(ROOT),str(COMFY),str(EXTRA),
            device,attention,"1" if preimport else "0"],text=True)
        return json.loads(value)

    def test_mps_sdpa_policy_is_not_silently_ignored(self):
        self.assertEqual(self.configure("mps","sdpa"),dict(cpu=False,gpu_only=True,sdpa=True))

    def test_default_attention_keeps_gpu_only_without_forcing_sdpa(self):
        self.assertEqual(self.configure("mps","default"),dict(cpu=False,gpu_only=True,sdpa=False))

    def test_cpu_reference_is_actually_cpu(self):
        self.assertEqual(self.configure("cpu","sdpa"),dict(cpu=True,gpu_only=False,sdpa=True))

    def test_preimported_policy_fails_instead_of_relabeling_it(self):
        self.assertIn("imported before",self.configure("mps","sdpa",preimport=True)["error"])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
    "real-video fixture tests require the validation interpreter")
class VideoFixtureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tc-seedvr2-real-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root/"source.mp4"
        self.video.write_bytes(b"mock-video-source")
        self.args = SimpleNamespace(video=self.video,output=self.root/"fixture",size=16,seed=42)

    def outputs(self,frames=5,jitter=False,truncated=False):
        info = dict(streams=[dict(width=16,height=16,nb_read_frames=str(frames),r_frame_rate="24/1",avg_frame_rate="24/1")])
        times = dict(frames=[dict(best_effort_timestamp_time=f"{i/24+(.01 if jitter and i==frames-1 else 0):.6f}")
            for i in range(frames)])
        raw = bytes([127])*(frames*16*16*3)
        return [json.dumps(info),json.dumps(times),raw[:-3] if truncated else raw]

    def run_fixture(self,**kwargs):
        with patch.object(oracle.shutil,"which",side_effect=lambda name:name):
            with patch.object(oracle.subprocess,"check_output",side_effect=self.outputs(**kwargs)):
                oracle.prepare_video(self.args)

    def test_complete_frames_and_noise_geometry_are_preserved(self):
        from safetensors.torch import load_file
        self.run_fixture()
        data = load_file(str(self.args.output/"input.safetensors"))
        self.assertEqual(tuple(data["pixels"].shape),(1,3,5,16,16))
        self.assertEqual(tuple(data["noise"].shape),(1,16,2,2,2))
        report = json.loads((self.args.output/"fixture.json").read_text())
        self.assertEqual((report["frames"],report["fps"]),(5,24))

    def test_truncated_decode_cannot_publish_a_fixture(self):
        with self.assertRaisesRegex(ValueError,"truncated"):
            self.run_fixture(truncated=True)
        self.assertFalse(self.args.output.exists())

    def test_vfr_with_matching_average_fps_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"timestamps"):
            self.run_fixture(jitter=True)
        self.assertFalse(self.args.output.exists())

    def test_incompatible_temporal_phase_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,"4n"):
            self.run_fixture(frames=4)
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
