#!/usr/bin/env python3
"""Negative bank-load tests; never change the qualified bank or load models."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(os.environ.get("TURBOCIDER_SEEDVR2_HYBRID_PROBE",ROOT / "build/seedvr2/seedvr2-hybrid-probe"))
MANIFEST = ROOT / "models/coreml/seedvr2-original-fp16-prefix2048-rows512-v1/manifest.json"


@unittest.skipUnless(PROBE.is_file() and MANIFEST.is_file(),"hybrid probe/qualified bank fixtures unavailable")
class ArtifactGuards(unittest.TestCase):
    def check_rejected(self, setup):
        with tempfile.TemporaryDirectory(prefix="seedvr2-artifact-rejection-") as directory:
            root = Path(directory)
            # Read-only source manifest; all writes target this test's scratch.
            manifest = root / "manifest.json"
            manifest.write_bytes(MANIFEST.read_bytes())
            for block in range(32):
                (root / f"block-{block:02}.mlmodelc").mkdir()
            setup(root,json.loads(manifest.read_text()))
            output = root / "must-not-exist"
            result = subprocess.run([str(PROBE),str(ROOT / "models/Comfy-Org-SeedVR2"),
                str(manifest),str(output),"512"],text=True,capture_output=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("artifact",result.stderr)
            self.assertFalse(output.exists(),"model construction/probe execution preceded artifact rejection")

    def test_missing_registered_files_are_rejected(self):
        self.check_rejected(lambda *_: None)

    def test_same_size_corrupted_registered_file_is_rejected(self):
        def corrupt(root, manifest):
            item = manifest["artifacts"]["0"]["identity"]["files"][0]
            target = root / "block-00.mlmodelc" / item["path"]
            target.parent.mkdir(parents=True,exist_ok=True)
            # Bound allocation; the first registered metadata file is small.
            self.assertLessEqual(item["bytes"],1 << 20)
            target.write_bytes(b"\0"*item["bytes"])
        self.check_rejected(corrupt)

    def test_symlink_entries_are_rejected(self):
        def link(root, manifest):
            (root / "block-00.mlmodelc" / "unsafe-link").symlink_to(MANIFEST)
        self.check_rejected(link)


if __name__ == "__main__": unittest.main()
