#!/usr/bin/env python3
"""CPU regression tests for the component observer's numerical gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("vae_cohort",ROOT/"tools/validation/seedvr2_vae_cohort.py")
cohort = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cohort)


@unittest.skipIf(np is None,"requires the NumPy/safetensors validation interpreter")
class VAEObservationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tc-seedvr2-vae-cohort-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root/"reference.safetensors"
        self.candidate = self.root/"candidate.safetensors"
        self.value = np.arange(32,dtype=np.float16).reshape(1,2,4,4)+1
        self.fixture = dict(latent=self.value,decoded=self.value/2)
        save_file(self.fixture,str(self.reference))

    def test_identical_outputs_pass_and_are_explicitly_bitwise_equal(self):
        save_file(self.fixture,str(self.candidate))
        rows = cohort.parity(self.reference,self.candidate)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(row["passed"] and row["bitwise_equal"] for row in rows))

    def test_finite_but_wrong_output_fails_without_loosened_thresholds(self):
        save_file(dict(latent=self.value,decoded=-self.value),str(self.candidate))
        rows = cohort.parity(self.reference,self.candidate)
        self.assertFalse(all(row["passed"] for row in rows))

    def test_nonfinite_result_fails_and_still_has_standard_json(self):
        broken = self.value.copy()
        broken.flat[0] = np.nan
        save_file(dict(latent=broken,decoded=self.value),str(self.candidate))
        rows = cohort.parity(self.reference,self.candidate)
        self.assertFalse(all(row["passed"] for row in rows))
        cohort.write_json(self.root/"report.json",dict(rows=rows))
        self.assertIsNone(next(row for row in json.loads((self.root/"report.json").read_text())["rows"]
            if not row["finite"])["cosine"])

    def test_shape_mismatch_is_rejected(self):
        save_file(dict(latent=self.value.reshape(-1),decoded=self.value),str(self.candidate))
        with self.assertRaisesRegex(ValueError,"shape"):
            cohort.parity(self.reference,self.candidate)

    def test_tensor_set_mismatch_is_rejected(self):
        save_file(dict(latent=self.value),str(self.candidate))
        with self.assertRaisesRegex(ValueError,"set"):
            cohort.parity(self.reference,self.candidate)

    def test_finalizer_cannot_modify_a_running_cohort(self):
        path = self.root/"report.json"
        cohort.write_json(path,dict(status="running",error="No module named 'safetensors'"))
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError,"only"):
            cohort.finalize_existing(SimpleNamespace(output=self.root))
        self.assertEqual(path.read_bytes(),original)

    def test_finalizer_cannot_recover_incomplete_gpu_observations(self):
        path = self.root/"report.json"
        identities = dict(probe="a"*64,input="b"*64,vae="c"*64)
        cohort.write_json(path,dict(status="failed",error="No module named 'safetensors'",
            identities=identities,order=["baseline","candidate","candidate","baseline"],cohorts=[]))
        original = path.read_bytes()
        args = SimpleNamespace(output=self.root,probe=self.reference,input=self.candidate,model=self.root)
        with patch.object(cohort,"sha256",side_effect=list(identities.values())):
            with self.assertRaisesRegex(ValueError,"incomplete"):
                cohort.finalize_existing(args)
        self.assertEqual(path.read_bytes(),original)
        self.assertFalse((self.root/"host-finalization-failure.json").exists())


if __name__ == "__main__":
    unittest.main()
