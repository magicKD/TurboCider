"""Bounded original-dtype capture and explicit Private numerical recipe gates."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tests.native.test_z_image_runtime_boundaries import CREATE_PROBE

ROOT=Path(__file__).resolve().parents[2]


class FfnCaptureHostTests(unittest.TestCase):
    def test_capture_preserves_dtype_and_has_process_cap_and_fresh_targets(self):
        source=(ROOT/"native/models/z_image/z_image.cpp").read_text()
        capture=source.split("void z_capture_runtime_ffn_input(",1)[1].split("struct RequestCacheLimit",1)[0]
        self.assertIn("captured>=config->limit",capture)
        self.assertIn("mx::contiguous(input)",capture)
        self.assertIn("input.dtype()==mx::bfloat16",capture)
        self.assertIn("input.shape(1)<=4224",capture)
        self.assertNotIn("mx::float16",capture)
        self.assertIn(".partial.safetensors",capture)
        self.assertIn("std::filesystem::rename(partial,path)",capture)
        self.assertIn('"runtime_build",runtime_build_identity()',capture)
        self.assertIn('"TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_LIMIT",1,32',source)

    def test_compact_policy_mil_and_configuration(self):
        with tempfile.TemporaryDirectory(prefix="tc-fp16-policy-") as folder:
            binary=Path(folder)/"policy"
            build=subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "tests/native/ane_fp16_value_config_test.cpp","native/backends/private/ane_mil.cpp","-o",str(binary)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn("PASS Private FP16 compact",run.stdout)
            print(run.stdout.strip())


@unittest.skipUnless(os.environ.get("TC_Z_FFN_CAPTURE_TEST_LIBRARY"),"explicit experimental library/local fixtures required")
class CaptureInitializationTests(unittest.TestCase):
    def test_controls_fail_before_capture_writes_and_accept_fresh_valid_config(self):
        library=str(Path(os.environ["TC_Z_FFN_CAPTURE_TEST_LIBRARY"]).resolve(strict=True))
        model=ROOT/"models/Comfy-Org-z_image_turbo"
        base={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        with tempfile.TemporaryDirectory(prefix="tc-ffn-capture-gates-") as raw:
            folder=Path(raw);used=folder/"used";used.mkdir();(used/"sentinel").write_text("keep")
            link=folder/"link";link.symlink_to(used,target_is_directory=True)
            fresh=folder/"unused";directory="TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_DIR"
            block="TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_BLOCK";limit="TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_LIMIT"
            cases=[({block:"2"},False),({limit:"1"},False),({directory:str(fresh)},False),
                ({directory:str(fresh),block:"32"},False),({directory:str(fresh),block:"-1"},False),
                ({directory:str(fresh),block:"2",limit:"0"},False),({directory:str(fresh),block:"2",limit:"33"},False),
                ({directory:str(used),block:"2"},False),({directory:str(link),block:"2"},False),
                ({directory:str(fresh),block:"2",limit:"8"},True)]
            for controls,accepted in cases:
                with self.subTest(controls=controls):
                    run=subprocess.run([sys.executable,"-c",CREATE_PROBE,library,str(model)],cwd=ROOT,
                        env={**base,**controls},capture_output=True,text=True,timeout=20)
                    self.assertEqual(run.returncode,0,run.stderr);row=json.loads(run.stdout)
                    self.assertEqual(row["status"]==0,accepted,row)
                    self.assertFalse(fresh.exists())
                    self.assertEqual((used/"sentinel").read_text(),"keep")


@unittest.skipUnless(os.environ.get("TC_ORDINARY_TEST_LIBRARY"),"explicit ordinary library/local fixtures required")
class CaptureOrdinaryBuildGateTests(unittest.TestCase):
    def test_capture_is_unavailable_in_ordinary_build(self):
        library=str(Path(os.environ["TC_ORDINARY_TEST_LIBRARY"]).resolve(strict=True))
        base={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        with tempfile.TemporaryDirectory(prefix="tc-ordinary-capture-") as raw:
            target=Path(raw)/"unused"
            run=subprocess.run([sys.executable,"-c",CREATE_PROBE,library,str(ROOT/"models/Comfy-Org-z_image_turbo")],
                cwd=ROOT,env={**base,"TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_DIR":str(target),
                    "TURBOCIDER_Z_RUNTIME_FFN_CAPTURE_BLOCK":"2"},capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr);row=json.loads(run.stdout)
            self.assertNotEqual(row["status"],0);self.assertIn("qe_capability_unqualified",row["error"])
            self.assertFalse(target.exists())


if __name__=="__main__":unittest.main(verbosity=2)
