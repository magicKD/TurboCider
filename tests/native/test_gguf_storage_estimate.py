"""Per-tensor storage prediction, not a whole-request memory claim."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_gguf_execution import fixture

ROOT=Path(__file__).resolve().parents[2]


class StorageEstimateTests(unittest.TestCase):
    def test_mixed_metadata_and_unsupported_affine(self):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-storage-") as raw:
            path=Path(raw);binary=path/"estimate";checkpoint=path/"mixed.gguf"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-I",str(ROOT/"native/core"),str(ROOT/"tools/validation/gguf_storage_estimate.cpp"),
                "-o",str(binary)],check=True)
            checkpoint.write_bytes(fixture([("q8",8,[64,3]),("q4",2,[64,3]),("q41",3,[64,3]),
                ("float",0,[3]),("half",1,[3]),("bf16",30,[3]),("k",12,[256])]))
            result=subprocess.run([str(binary),str(checkpoint)],check=True,capture_output=True,text=True)
            report=json.loads(result.stdout);self.assertEqual(report["tensor_count"],7)
            types={row["type"]:row for row in report["types"]}
            for typ,source,affine in ((8,204,216),(2,108,120),(3,120,120),(0,12,12),(1,6,6),(30,6,6)):
                self.assertEqual(types[typ]["source_payload_bytes"],source)
                self.assertEqual(types[typ]["native_affine_payload_bytes"],affine)
                self.assertEqual(types[typ]["native_affine_capacity_upper_bytes"],3*16384 if typ in (2,3,8) else 16384)
            self.assertEqual(types[8]["logical_bf16_bytes"],384)
            self.assertIsNone(types[12]["native_affine_payload_bytes"])
            self.assertIsNone(types[12]["native_affine_capacity_upper_bytes"])
            checkpoint.write_bytes(checkpoint.read_bytes()[:-1])
            self.assertNotEqual(subprocess.run([str(binary),str(checkpoint)],capture_output=True).returncode,0)


if __name__=="__main__":unittest.main()
