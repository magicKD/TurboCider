"""Portable raw-array checks, including numeric-equal but different zero bits."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_lora_swiglu_block_audit as audit


class SwigluBlockAuditTests(unittest.TestCase):
    def write(self,path,words,shape=None):
        shape=shape or [len(words)]
        payload=struct.pack("<"+"H"*len(words),*words)
        header=json.dumps(dict(tensor=dict(dtype="BF16",shape=shape,data_offsets=[0,len(payload)]))).encode()
        path.write_bytes(struct.pack("<Q",len(header))+header+payload)

    def test_full_raw_bits_are_not_only_floats(self):
        with tempfile.TemporaryDirectory(prefix="tc-swiglu-audit-") as directory:
            a=Path(directory)/"a.safetensors";b=Path(directory)/"b.safetensors"
            self.write(a,[0,0x3f80]);self.write(b,[0x8000,0x3f80])
            with self.assertRaises(ValueError):audit.same(audit.arrays(a)["tensor"],audit.arrays(b)["tensor"])
            audit.same(audit.arrays(a)["tensor"],audit.arrays(a)["tensor"])

    def test_nan_infinity_and_incomplete_payload_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="tc-swiglu-audit-") as directory:
            path=Path(directory)/"a.safetensors"
            for words in ([0x7f80],[0xff80],[0x7fc0]):
                self.write(path,words)
                with self.subTest(words=words),self.assertRaises(ValueError):audit.arrays(path)
            self.write(path,[0x3f80],[2])
            with self.assertRaises(ValueError):audit.arrays(path)


if __name__=="__main__":unittest.main()
