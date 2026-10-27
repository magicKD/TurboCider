"""Host emitter golden/row-family checks; no ANE or model execution."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
# Captured before the family change using the V6 native emitter and exact V6
# source archive. Covers real Qwen shapes and tiny base/LoRA/Comfy recipes.
GOLDEN={
    '0.bin':'429504f5fe3a5b2669104c7b6c0d001f9fb6276c21b7fee9a1ddb7946567402a',
    '0.mil':'3e02e6966bbab710d798388fb50be9c19114cbc9c37a0e27916cd06c61e53ba6',
    '1.bin':'429504f5fe3a5b2669104c7b6c0d001f9fb6276c21b7fee9a1ddb7946567402a',
    '1.mil':'e27a86a7622f080b0ca0234fb8a63dfb8abd84656c95bcb6fce1d1d48174cf06',
    '2.bin':'8f64d281da7f50dde8175ceb144a778c0d579d9141f23c0aff2b1b28dbed6849',
    '2.mil':'954a024ab6cd6c28a397df42383b969b1637fcfec4e9642a8def39326cec5664',
    '3.bin':'8f64d281da7f50dde8175ceb144a778c0d579d9141f23c0aff2b1b28dbed6849',
    '3.mil':'dfee8006dc57c31288af901a0636a98a08464cd9795f6bcafaacf474477c6681',
    '4.bin':'ac5ee01d7c130715aef3f0cb50ab8e0eacab9f043dbf69ed44e21c21a1df1cdf',
    '4.mil':'972034346635c5b53db044962eaa74952d78af2bfa7aeb1af68867be1d96f3fa',
}


class W8FunctionEmitterTests(unittest.TestCase):
    def test_family_and_unchanged_legacy_bytes(self):
        with tempfile.TemporaryDirectory(prefix='tc-w8-functions-') as d:
            out=Path(d);binary=out/'probe'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',
                str(ROOT/'tests/native/ane_w8_functions_test.cpp'),str(ROOT/'native/backends/private/ane_mil.cpp'),
                '-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary),str(out)],check=True,cwd=ROOT)
            for name,expected in GOLDEN.items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),expected,name)


if __name__=='__main__':unittest.main()
