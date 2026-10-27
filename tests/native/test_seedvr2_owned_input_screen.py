#!/usr/bin/env python3
"""CPU guards for terminal qualification and exact ownership output checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "tools/validation"))
import seedvr2_owned_input_screen as screen


class TerminalGuards(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="seedvr2-owned-screen-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sha = "a"*64
        self.report = self.root / "cohort.json"
        order = ["native-gpu","comfy-default","comfy-default","native-gpu"]
        rows = []
        for index, route in enumerate(order):
            path = self.root / f"child-{index}.json"
            path.write_text(json.dumps(dict(status="complete",runs=[{}]*4,server_terminal=True,
                native_artifact_identity=dict(library_sha256=self.sha))))
            rows.append(dict(index=index,route=route,report=str(path)))
        self.record = dict(status="complete",speed_gate_passed=True,order=order,cohorts=rows,library_sha256=self.sha)

    def check(self):
        self.report.write_text(json.dumps(self.record))
        return screen.completed_cohort(self.report,self.sha)

    def test_complete_bound_terminal_gate_is_accepted(self):
        self.assertEqual(self.check()["status"],"complete")

    def test_running_failed_partial_or_wrong_library_cannot_start_screen(self):
        for key, value in (("status","running"),("speed_gate_passed",False),("library_sha256","b"*64),("cohorts",self.record["cohorts"][:3])):
            original = self.record[key]
            self.record[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError): self.check()
            self.record[key] = original

    def test_live_reference_and_wrong_native_child_are_rejected(self):
        reference = Path(self.record["cohorts"][1]["report"])
        child = json.loads(reference.read_text()); child["server_terminal"] = False
        reference.write_text(json.dumps(child))
        with self.assertRaises(ValueError): self.check()
        child["server_terminal"] = True; reference.write_text(json.dumps(child))
        native = Path(self.record["cohorts"][0]["report"])
        child = json.loads(native.read_text()); child["native_artifact_identity"]["library_sha256"] = "bad"
        native.write_text(json.dumps(child))
        with self.assertRaises(ValueError): self.check()


class ExactTensorGuards(unittest.TestCase):
    def test_signed_zero_difference_is_not_called_bitwise_equal(self):
        import numpy as np
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory(prefix="seedvr2-owned-tensors-") as directory:
            root = Path(directory)
            arrays = {name:np.ones((1,2,3),dtype=np.float16) for name in ("pixels","condition","velocity","latent")}
            arrays["pixels"][0,0,0] = 0
            save_file(arrays,str(root / "left.safetensors"))
            save_file(arrays,str(root / "right.safetensors"))
            self.assertTrue(all(row["bitwise_equal"] for row in screen.exact_tensors(root / "left.safetensors",root / "right.safetensors")))
            arrays["pixels"][0,0,0] = -0.
            save_file(arrays,str(root / "right.safetensors"))
            self.assertFalse(next(row for row in screen.exact_tensors(root / "left.safetensors",root / "right.safetensors") if row["tensor"] == "pixels")["bitwise_equal"])
            arrays["pixels"][0,0,0] = np.nan
            save_file(arrays,str(root / "right.safetensors"))
            self.assertFalse(all(row["finite"] for row in screen.exact_tensors(root / "left.safetensors",root / "right.safetensors")))


if __name__ == "__main__": unittest.main()
