"""Host-only process ownership and memory-evidence integration contracts."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
with mock.patch.object(sys, "path", [str(ROOT / "tools/validation"), *sys.path]):
    import runtime_ane_memory as MEMORY


class MemoryScreenTests(unittest.TestCase):
    def test_exact_owned_child_role_and_wildcard_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tools/native").mkdir(parents=True)
            for name in ("process_tree_sampler.py", "verify_process_tree_samples.py", "reference"):
                (root / ("reference" if name == "reference" else "tools/native/" + name)).write_text("fixture")
            raw, report = root / "child-memory.jsonl", root / "child-memory-report.json"
            def run(command, **kwargs):
                self.assertEqual(command[command.index("--role") + 1], "reference_inference=" + str((root / "reference").resolve()))
                self.assertEqual(command[-2:], ["--", "fixture"])
                raw.write_text("raw fixture evidence\n")
                report.write_text(json.dumps(dict(complete=True, command_exit_code=0, allowed_max_gap_ns=500_000_000,
                    correlation_id=command[command.index("--correlation-id") + 1])))
                return subprocess.CompletedProcess(command, 0)
            args = dict(repo=root, output=root, stem="child", env={}, stdout=None, stderr=None,
                        timeout=30, interval_ms=100, max_gap_ms=500)
            with mock.patch.object(MEMORY, "run_owned", side_effect=run) as launch, \
                 mock.patch.object(MEMORY.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                MEMORY.run_sampled(["fixture"], child_roles=(("reference_inference", str(root / "reference")),), **args)
                self.assertEqual(launch.call_count, 1)
                for role, path in (("reference_inference", "*"), ("reference_inference", "/tmp/*"),
                                   ("reference_inference", "reference"), ("", str(root / "reference"))):
                    with self.assertRaises(ValueError):
                        MEMORY.run_sampled(["fixture"], child_roles=((role, path),), **args)
                self.assertEqual(launch.call_count, 1)

    def test_nonmain_caller_rejected_before_any_process_launch(self):
        with mock.patch.object(MEMORY.threading, "current_thread", return_value=object()), \
                mock.patch.object(MEMORY.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "main thread"):
                MEMORY.run_owned(["fixture"], timeout=9)
            launch.assert_not_called()

    def test_owned_success_uses_private_session_without_signalling(self):
        process = mock.Mock(pid=1234)
        process.wait.return_value = 0
        with mock.patch.object(MEMORY.subprocess, "Popen", return_value=process) as launch, \
                mock.patch.object(MEMORY.os, "killpg") as kill:
            self.assertEqual(MEMORY.run_owned(["fixture"], timeout=9).returncode, 0)
        launch.assert_called_once_with(["fixture"], start_new_session=True)
        process.wait.assert_called_once_with(timeout=9)
        kill.assert_not_called()

    def test_timeout_and_interrupt_drain_only_owned_group_and_preserve_error(self):
        for error in (subprocess.TimeoutExpired("fixture", 9), KeyboardInterrupt(),
                      InterruptedError("SIGTERM")):
            for term_exits in (True, False):
                process = mock.Mock(pid=1234)
                process.wait.side_effect = [error, 0 if term_exits else
                                           subprocess.TimeoutExpired("fixture", 5), 0]
                before = signal.getsignal(signal.SIGTERM)
                with self.subTest(error=type(error), term_exits=term_exits), \
                        mock.patch.object(MEMORY.subprocess, "Popen", return_value=process), \
                        mock.patch.object(MEMORY.os, "killpg") as kill:
                    with self.assertRaises(type(error)) as caught:
                        MEMORY.run_owned(["fixture"], timeout=9)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(kill.call_args_list, [mock.call(1234, signal.SIGTERM),
                                                          mock.call(1234, signal.SIGKILL)])
                    self.assertEqual(signal.getsignal(signal.SIGTERM), before)
                    self.assertEqual(process.wait.call_args_list,
                                     [mock.call(timeout=9), mock.call(timeout=5), mock.call(timeout=5)])

    def test_crashed_wrapper_cleans_descendants_and_missing_group_is_safe(self):
        process = mock.Mock(pid=1234)
        process.wait.return_value = -9
        with mock.patch.object(MEMORY.subprocess, "Popen", return_value=process), \
                mock.patch.object(MEMORY.os, "killpg", side_effect=ProcessLookupError) as kill:
            self.assertEqual(MEMORY.run_owned(["fixture"], timeout=9).returncode, -9)
            kill.assert_called_once_with(1234, signal.SIGKILL)

    def test_memory_receipt_and_verifier_fail_closed_with_raw_evidence_retained(self):
        cases = [({}, 0, 0, None), ({}, 3, 0, RuntimeError), ({}, 0, 3, RuntimeError),
                 ({"complete": False}, 0, 0, ValueError),
                 ({"correlation_id": "unrelated"}, 0, 0, ValueError),
                 ({"command_exit_code": True}, 0, 0, ValueError),
                 ({"allowed_max_gap_ns": 999}, 0, 0, ValueError)]
        for change, sampler_exit, verifier_exit, error in cases:
            with self.subTest(change=change, sampler=sampler_exit, verifier=verifier_exit), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "tools/native").mkdir(parents=True)
                for name in ("process_tree_sampler.py", "verify_process_tree_samples.py"):
                    (root / "tools/native" / name).write_text("fixture")
                raw, report = root / "0-gpu-memory.jsonl", root / "0-gpu-memory-report.json"
                def run(command, **kwargs):
                    raw.write_text("raw fixture evidence\n")
                    self.assertEqual(command[-5:], ["--", "cli", "batch", "model", "request"])
                    correlation = command[command.index("--correlation-id") + 1]
                    verified = {"complete": True, "correlation_id": correlation,
                                "command_exit_code": 0, "allowed_max_gap_ns": 500_000_000}
                    report.write_text(json.dumps({**verified, **change}))
                    return subprocess.CompletedProcess(command, sampler_exit)
                with mock.patch.object(MEMORY, "run_owned", side_effect=run), \
                        mock.patch.object(MEMORY.subprocess, "run", return_value=
                                          subprocess.CompletedProcess([], verifier_exit)) as verify:
                    args = dict(repo=root, output=root, stem="0-gpu", env={}, stdout=None,
                                stderr=None, timeout=30, interval_ms=100, max_gap_ms=500)
                    if error:
                        with self.assertRaises(error):
                            MEMORY.run_sampled(["cli", "batch", "model", "request"], **args)
                    else:
                        receipt = MEMORY.run_sampled(["cli", "batch", "model", "request"], **args)
                        self.assertEqual(receipt["evidence"], raw.name)
                        self.assertEqual(receipt["evidence_sha256"], MEMORY.sha256(raw))
                        self.assertEqual(receipt["report_sha256"], MEMORY.sha256(report))
                        self.assertEqual(len(receipt["tool_sha256"]), 2)
                        self.assertIn("load+cold+warm+exit", receipt["scope"])
                    self.assertTrue(raw.exists())
                    self.assertTrue(report.exists())
                    verify.assert_called_once()

    def test_real_timeout_stops_owned_parent_and_child(self):
        # Small standard-library processes only, no sampler or GPU. The child
        # ignores TERM to cover a wrapper exiting before its descendant.
        code = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])
print(child.pid, flush=True)
time.sleep(30)
"""
        with tempfile.TemporaryFile(mode="w+") as output:
            with self.assertRaises(subprocess.TimeoutExpired):
                MEMORY.run_owned([sys.executable, "-c", code], stdout=output, timeout=0.5)
            output.seek(0)
            child_pid = int(output.read().strip())
        state = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="],
                               capture_output=True, text=True)
        # A reparented zombie is terminated, not a running leaked workload.
        self.assertTrue(state.returncode != 0 or state.stdout.strip().startswith("Z"), state.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
