#!/usr/bin/env python3
"""Revision-runner preflight guards; no models or GPU execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/validation/seedvr2_native_revision_cohort.py"


class RevisionPreflightTests(unittest.TestCase):
    def command(self, *extra):
        return [sys.executable,str(RUNNER),"--model","unused-model","--source","unused-source",
            "--before-library","unused-before","--after-library","unused-after",
            "--output","unused-output","--kind","image","--size","1024",*extra]

    def test_short_cohorts_are_rejected_before_reading_artifacts(self):
        result = subprocess.run(self.command("--count","3"),text=True,capture_output=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("at least three genuine warm",result.stderr)

    def test_image_frame_mismatch_is_rejected_before_reading_artifacts(self):
        result = subprocess.run(self.command("--frames","2"),text=True,capture_output=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("one cold",result.stderr)

    def test_existing_cohort_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="seedvr2-revision-preflight-") as directory:
            root = Path(directory)
            artifact = root / "fixture"
            artifact.write_bytes(b"not-a-library")
            output = root / "existing"
            output.mkdir()
            marker = output / "original"
            marker.write_bytes(b"preserve")
            command = self.command("--model",str(root),"--source",str(artifact),
                "--before-library",str(artifact),"--after-library",str(artifact),"--output",str(output))
            result = subprocess.run(command,text=True,capture_output=True)
            self.assertEqual(result.returncode,2)
            self.assertIn("fresh revision cohort",result.stderr)
            self.assertEqual(marker.read_bytes(),b"preserve")
            self.assertEqual(sorted(path.name for path in output.iterdir()),["original"])


if __name__ == "__main__": unittest.main()
