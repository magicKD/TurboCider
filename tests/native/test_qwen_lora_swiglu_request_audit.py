"""Portable raw finite and exact request-normalization checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/validation"))
import qwen_lora_swiglu_request_audit as audit


class SwigluRequestAuditTests(unittest.TestCase):
    def test_bf16_zero_signs_are_finite_but_nan_and_inf_rejected(self):
        audit.finite_raw(("BF16",np.asarray([0,0x8000,0x3f80,0x7f7f],dtype=np.uint16)))
        for value in (0x7f80,0xff80,0x7fc1):
            with self.subTest(value=value),self.assertRaises(ValueError):audit.finite_raw(("BF16",np.asarray([value],dtype=np.uint16)))
        with self.assertRaises(ValueError):audit.finite_raw(("F32",np.asarray([float("nan")],dtype=np.float32)))

    def test_request_normalization_removes_only_observer_paths(self):
        value=dict(output="a.png",dump_tensors="raw",steps=5,prompt="same",inputs=[dict(path="ref.png")])
        with tempfile.TemporaryDirectory(prefix="tc-swiglu-request-audit-") as directory:
            path=Path(directory)/"request.json";path.write_text(json.dumps(value))
            self.assertEqual(audit.request(path),dict(steps=5,prompt="same",inputs=[dict(path="ref.png")]))


if __name__=="__main__":unittest.main()
