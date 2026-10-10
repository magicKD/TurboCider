"""Actual native MLX field/projection oracle and packed-bank ownership negatives."""
import os
from pathlib import Path
import struct
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


def fixture(entries):
    header=struct.pack("<IIQQ",0x46554747,3,len(entries),0);payload=bytearray()
    for name,typ,dims,data in entries:
        raw=name.encode();payload.extend(bytes((-len(payload))%32))
        header+=struct.pack("<Q",len(raw))+raw+struct.pack("<I",len(dims))
        header+=b"".join(struct.pack("<Q",dim) for dim in dims)+struct.pack("<IQ",typ,len(payload))
        payload.extend(data)
    return header+bytes((-len(header))%32)+payload


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual Metal opt-in required")
class PackedBankTests(unittest.TestCase):
    def test_native_oracle_and_ownership(self):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-bank-") as raw:
            folder=Path(raw);q8=struct.pack("<e",.125)+bytes((i*17)%256 for i in range(32))
            q4=struct.pack("<e",-.25)+bytes((i*19)%256 for i in range(16))
            q41=struct.pack("<ee",.25,-.125)+bytes((i*23)%256 for i in range(16))
            good=folder/"fixture.gguf"
            good.write_bytes(fixture([("q8.weight",8,[64,512],q8*1024),
                ("q4.weight",2,[64,512],q4*1024),("q41.weight",3,[64,512],q41*1024),
                ("f32",0,[8],struct.pack("<8f",0.,-0.,1.,-1.,.125,-.5,3.5,2.)),
                ("f16",1,[8],struct.pack("<8e",0.,-0.,1.,-1.,.125,-.5,3.5,2.)),
                ("bf16",30,[8],struct.pack("<8H",0,0x8000,0x3f80,0xbf80,0x3eab,0x0001,0x477f,0x3fff))]))
            bad=folder/"nonfinite.gguf";bad.write_bytes(fixture([("bad",30,[8],struct.pack("<8H",*[0x7f80]*8))]))
            collision=folder/"collision.gguf";collision.write_bytes(fixture([
                ("q8.weight",8,[64,1],q8*2),("q8.scales",0,[1],struct.pack("<f",1.))]))
            unsupported=folder/"unsupported.gguf";unsupported.write_bytes(fixture([("k.weight",12,[256,1],bytes(144))]))
            row=folder/"row-floor.gguf";row.write_bytes(fixture([("large",0,[4097],bytes(4097*4))]))
            q4k=struct.pack("<ee",.125,.0625)+bytes((i*13+3)%256 for i in range(12))+bytes((i*17)%256 for i in range(128))
            q5k=struct.pack("<ee",.125,.0625)+bytes((i*13+3)%256 for i in range(12))+bytes((i*19)%256 for i in range(32))+bytes((i*17)%256 for i in range(128))
            q6k=bytes((i*17)%256 for i in range(128))+bytes((i*19)%256 for i in range(64))+bytes((i*23+1)%256 for i in range(16))+struct.pack("<e",.015625)
            mixed=folder/"mixed-k.gguf";mixed.write_bytes(fixture([
                ("q4k.weight",12,[512,37],q4k*74),("q5k.weight",13,[512,37],q5k*74),
                ("q6k.weight",14,[512,37],q6k*74)]))
            nonfinite_k=folder/"nonfinite-k.gguf";nonfinite_k.write_bytes(fixture([
                ("q4k.weight",12,[512,37],q4k*74),("bad.weight",14,[512,37],(q6k[:-2]+struct.pack("<e",float("inf")))*74)]))
            mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
            library=ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/quantized-execution")
            binary=folder/"probe"
            flags=["-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=15.0"]
            sanitizer=os.environ.get("TC_STREAMING_SANITIZER")
            if sanitizer: flags+=["-fsanitize="+sanitizer,"-fno-omit-frame-pointer"]
            subprocess.run(["xcrun","clang++",*flags,"-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),
                "-isystem",str(mlx/"include"),str(ROOT/"tests/native/gguf_packed_bank_test.cpp"),
                str(ROOT/"native/runtime/streaming/gguf_packed_bank.cpp"),str(ROOT/"native/core/gguf_decode.cpp"),
                str(ROOT/"native/core/gguf_affine.cpp"),"-L",str(library),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(library),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(binary)],check=True)
            result=subprocess.run([str(binary),str(good),str(bad),str(collision),str(unsupported),str(row),str(mixed),str(nonfinite_k)],
                                  capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS GGUF packed bank Metal",result.stdout);print(result.stdout,end="")


if __name__=="__main__":unittest.main()
