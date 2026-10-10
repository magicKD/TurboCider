"""ConvRot observer/recipe errors must reject before loading a device library."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ConvRotProbeOptionsTests(unittest.TestCase):
    def reject(self,options,message,environment=None):
        with tempfile.TemporaryDirectory(prefix="tc-convrot-options-") as raw:
            output=Path(raw)/"out"
            result=subprocess.run([sys.executable,str(ROOT/"tools/native/run_z_image_convrot_runtime.py"),
                "--library","missing.dylib","--model",raw,"--checkpoint","missing.safetensors","--output",str(output),
                *options],capture_output=True,text=True,env=environment,timeout=10)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn(message,result.stderr);self.assertFalse(output.exists())

    def test_diagnostics_and_route_are_explicit(self):
        for mode in ("timing","memory"):
            self.reject(["--route","gpu","--gpu-recipe","compiled_dense","--measurement",mode,"--dump"],"requires diagnostic")
            self.reject(["--route","gpu","--gpu-recipe","compiled_dense","--measurement",mode,"--validate-source-blocks"],"requires diagnostic")
            self.reject(["--route","gpu","--measurement",mode],"cannot contaminate",dict(os.environ,TURBOCIDER_Z_CONVROT_VALIDATE_BLOCKS="1"))
        self.reject(["--route","gpu","--validate-source-blocks"],"requires compiled ConvRot")
        self.reject(["--route","runtime","--manifest","missing","--gpu-recipe","compiled_butterfly"],"requires GPU route")
        self.reject(["--route","gpu","--warmup","9"],"warmup 0..8")
        self.reject(["--route","gpu","--gpu-cache-bytes","1"],"cache hint requires compiled GPU recipe")
        self.reject(["--route","gpu","--gpu-recipe","compiled_dense","--gpu-cache-bytes",str((4<<30)+1)],"cache hint requires")


if __name__=="__main__":unittest.main()
