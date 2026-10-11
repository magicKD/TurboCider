import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from runtime_ane_memory import run_sampled


@unittest.skipUnless(sys.platform == "darwin", "libproc process-tree counters require Darwin")
class AsyncEvidenceLiveTests(unittest.TestCase):
    def test_real_owned_cpu_process_and_durable_independent_replay(self):
        with tempfile.TemporaryDirectory(prefix="tc-live-async-evidence-") as raw:
            output = Path(raw)
            code = "import time; value=bytearray(16<<20); time.sleep(.45); assert len(value)==16<<20"
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                receipt = run_sampled([sys.executable, "-c", code], repo=ROOT, output=output, stem="cpu",
                    env={}, stdout=out, stderr=err, timeout=20, interval_ms=50, max_gap_ms=500, async_evidence=True)
            self.assertIs(receipt["verified"]["complete"], True)
            self.assertEqual(receipt["verified"]["command_exit_code"], 0)
            self.assertEqual(receipt["verified"]["evidence_writer_policy"], "bounded-async-record-fsync-v1")
            self.assertGreaterEqual(receipt["verified"]["sample_count"], 3)
            self.assertEqual(receipt["verified"]["allowed_max_gap_ns"], 500_000_000)
            replay = json.loads(subprocess.check_output([sys.executable, str(ROOT / "tools/native/verify_process_tree_samples.py"),
                str(output / "cpu-memory.jsonl")], text=True))
            self.assertEqual(replay, receipt["verified"])


if __name__ == "__main__":
    unittest.main()
