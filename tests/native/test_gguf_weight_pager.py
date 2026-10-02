"""Actual Metal/StageExecutor integration, explicitly enabled; no mock hardware pass."""
import os
from pathlib import Path
import struct
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
NATIVE_LIBRARY_DIR=ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR", "build/native")

@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1", "explicit Metal opt-in required")
class GgufPagerTests(unittest.TestCase):
    def test_native_slots_gpu_and_ledger(self):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-pager-") as raw:
            folder=Path(raw); model=folder/"fixture.gguf"; binary=folder/"probe"
            def string(s):
                data=s.encode(); return struct.pack("<Q",len(data))+data
            header=struct.pack("<IIQQ",0x46554747,3,9,0); payload=bytearray()
            for i in range(4):
                payload.extend(bytes((-len(payload))%32)); offset=len(payload)
                header+=string(f"layers.{i}.weight")+struct.pack("<IQQIQ",2,64,512,8,offset)
                payload.extend((struct.pack("<e",1.)+bytes([i+1]*32))*1024)
            payload.extend(bytes((-len(payload))%32)); offset=len(payload)
            header+=string("embedding.weight")+struct.pack("<IQQIQ",2,64,4,8,offset)
            for row in range(4): payload.extend((struct.pack("<e",1.)+bytes([row+1]*32))*2)
            payload.extend(bytes((-len(payload))%32)); offset=len(payload)
            header+=string("fixed")+struct.pack("<IQIQ",1,1,0,offset)
            payload.extend(struct.pack("<f",3.5))
            for typ in (0,1,30):
                payload.extend(bytes((-len(payload))%32));offset=len(payload)
                header+=string(f"fixed{typ}")+struct.pack("<IQQIQ",2,64,257,typ,offset)
                for i in range(64*257):
                    value=(-1 if i%2 else 1)*(1+(i%4)*2**-8+2**-9)
                    payload.extend(struct.pack("<f" if typ==0 else "<e",value) if typ!=30 else
                        struct.pack("<H",0x3f80+i%4))
            header+=bytes((-len(header))%32)
            model.write_bytes(header+payload)
            library=Path(sysconfig.get_paths()["purelib"])/"mlx"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=15.0","-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),"-isystem",str(library/"include"),
                str(ROOT/"tests/native/gguf_weight_pager_test.cpp"),str(ROOT/"native/runtime/streaming/gguf_weight_pager.cpp"),
                str(ROOT/"native/core/gguf_decode.cpp"),"-L",str(NATIVE_LIBRARY_DIR),"-lturbocider",
                str(ROOT/"native/core/gguf_affine.cpp"),
                "-L",str(library/"lib"),"-lmlx","-Wl,-rpath,"+str(NATIVE_LIBRARY_DIR),
                "-Wl,-rpath,"+str(library/"lib"),"-o",str(binary)],check=True)
            result=subprocess.run([str(binary),str(model),"--mutate-owned-fixture"],text=True,capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS GGUF pager Metal",result.stdout)
            print(result.stdout,end="")

if __name__=="__main__": unittest.main()
