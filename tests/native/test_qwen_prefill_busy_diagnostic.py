import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
SPEC=importlib.util.spec_from_file_location("qwen_busy_diagnostic",ROOT/"tools/validation/qwen_prefill_busy_diagnostic.py")
DIAGNOSTIC=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(DIAGNOSTIC)


class BusyDiagnosticTests(unittest.TestCase):
    def test_policy_is_explicit_and_cannot_silently_replace_strict_load_screen(self):
        args=["--cli","unused","--output","unused"]
        self.assertEqual(DIAGNOSTIC.screen_arguments(["--",*args]),["--defer-prefill-screen","--sample-memory",*args])
        self.assertEqual(args,["--cli","unused","--output","unused"])
        self.assertEqual(DIAGNOSTIC.screen_arguments([*args,"--gpu-first-prefill-screen"]),
            ["--sample-memory",*args,"--gpu-first-prefill-screen"])
        self.assertEqual(DIAGNOSTIC.screen_arguments([*args,"--compiled-encoder-screen"]),
            ["--sample-memory",*args,"--compiled-encoder-screen"])
        self.assertEqual(DIAGNOSTIC.screen_arguments(args,True),["--prefill-layer-screen","--sample-memory",*args])
        self.assertEqual(DIAGNOSTIC.screen_arguments(args,phases=True),["--sample-memory",*args])
        with self.assertRaises(ValueError):DIAGNOSTIC.screen_arguments(args,True,True)
        for bad in ([],["--"],[*args,"--observe-load"],[*args,"--defer-prefill-screen"],[*args,"--prefill-layer-screen"],[*args,"--sample-memory"]):
            with self.assertRaises(ValueError):DIAGNOSTIC.screen_arguments(bad)

    def test_strict_flag_rejected_before_subprocess_or_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);load=root/"load.jsonl";report=root/"report.json"
            result=subprocess.run([sys.executable,"-S",str(ROOT/"tools/validation/qwen_prefill_busy_diagnostic.py"),
                "--load-evidence",str(load),"--report",str(report),"--","--observe-load"],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertFalse(load.exists());self.assertFalse(report.exists())


if __name__=="__main__":unittest.main()
