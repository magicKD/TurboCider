"""Reference-helper tests never access the network or execute upstream code."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_references", ROOT / "scripts/fetch_references.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "references"
        self.project = {"id": "R01", "repository": "example/research", "checkout": "research"}

    def test_plan_does_not_create_root_or_run_git(self):
        with mock.patch.object(fetch, "run") as run:
            self.assertEqual(fetch.checkout(self.project, self.root)["status"], "planned_not_downloaded")
        run.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_existing_non_git_directory_is_preserved(self):
        target = self.root / "research"
        target.mkdir(parents=True)
        (target / "important.txt").write_text("preserve me")
        with mock.patch.object(fetch, "run") as run:
            result = fetch.checkout(self.project, self.root, download=True)
        self.assertEqual(result["status"], "existing_non_git_preserved")
        self.assertEqual((target / "important.txt").read_text(), "preserve me")
        run.assert_not_called()

    def test_existing_checkout_is_read_only_even_when_download_requested(self):
        (self.root / "research/.git").mkdir(parents=True)
        replies = [SimpleNamespace(returncode=0, stdout="abc123\n"), SimpleNamespace(returncode=0, stdout=" M file\n")]
        with mock.patch.object(fetch, "run", side_effect=replies) as run:
            result = fetch.checkout(self.project, self.root, download=True)
        self.assertEqual(result["status"], "existing_preserved")
        self.assertTrue(result["tracked_worktree_dirty"])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertIn("--no-optional-locks", call.args[0])
            self.assertIn("core.fsmonitor=false", call.args[0])
            self.assertFalse(set(call.args[0]) & {"clone", "fetch", "reset", "clean", "checkout"})

    def test_unsafe_inventory_fields_are_rejected_before_git(self):
        for field, value in (("checkout", "../escape"), ("checkout", []),
                             ("repository", "https://other.invalid/repo"), ("repository", "owner/../repo"),
                             ("sparse_paths", ["../escape"]), ("sparse_paths", ["--help"]),
                             ("sparse_paths", [{}]), ("branch", "--upload-pack=bad"), ("branch", "a/../b")):
            project = {**self.project, field: value}
            with self.subTest(field=field, value=value), mock.patch.object(fetch, "run") as run:
                with self.assertRaises(ValueError):
                    fetch.checkout(project, self.root)
                run.assert_not_called()

    def test_symlink_target_is_rejected(self):
        self.root.mkdir()
        (self.root / "research").symlink_to(Path(self.temporary.name), target_is_directory=True)
        with self.assertRaises(ValueError):
            fetch.checkout(self.project, self.root)

    def test_failed_clone_is_reported_without_followup_commands(self):
        with mock.patch.object(fetch, "run", return_value=SimpleNamespace(returncode=1, stderr="clone failed")) as run:
            result = fetch.checkout(self.project, self.root, download=True)
        self.assertEqual(result["status"], "clone_failed")
        self.assertEqual(run.call_count, 1)
        self.assertIn("https://github.com/example/research.git", run.call_args.args[0])

    def test_inventory_entries_are_unique_and_plan_safely(self):
        projects = json.loads((ROOT / "candidates/projects.json").read_text())["projects"]
        self.assertEqual(len({item["id"] for item in projects}), len(projects))
        self.assertEqual(len({item["checkout"] for item in projects}), len(projects))
        with mock.patch.object(fetch, "run") as run:
            for project in projects:
                self.assertEqual(fetch.checkout(project, self.root)["status"], "planned_not_downloaded")
        run.assert_not_called()

    def test_cli_plan_stdout_is_machine_readable_json(self):
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/fetch_references.py"),
                                "--project", "R01", "--reference-root", str(self.root)],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertFalse(json.loads(child.stdout)["network_explicitly_enabled"])
        self.assertIn("Inspecting R01", child.stderr)
        self.assertFalse(self.root.exists())

    def test_timeout_produces_failure_receipt_not_traceback(self):
        arguments = ["fetch_references.py", "--project", "R01", "--reference-root", str(self.root), "--download"]
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(fetch, "checkout", side_effect=subprocess.TimeoutExpired("git", 180)), \
                mock.patch("sys.stdout") as stdout, mock.patch("sys.stderr"):
            self.assertEqual(fetch.main(), 1)
        report = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
        self.assertEqual(report["results"][0]["status"], "operation_failed")


if __name__ == "__main__":
    unittest.main()
