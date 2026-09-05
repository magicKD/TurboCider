from __future__ import annotations

import os
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from turbocider.adapters.base import CommandSpec
from turbocider.jobs import JobManager
from turbocider.models import GenerationRequest, JobState


class JobManagerTests(unittest.TestCase):
    def test_completed_job_is_restored_after_daemon_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = JobManager(root)
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            record = first.submit(
                request,
                "test-plan",
                CommandSpec(argv=["/usr/bin/true"], cwd=root),
            )
            record = first.wait(record.id, timeout=5)
            first.close()

            second = JobManager(root)
            try:
                restored = second.get(record.id)
                self.assertEqual(restored.state, JobState.SUCCEEDED)
                self.assertEqual(restored.plan_id, "test-plan")
            finally:
                second.close()

    def test_active_job_is_marked_interrupted_after_daemon_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            jobs.mkdir(parents=True)
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            record = {
                "id": "stale",
                "request": request.as_dict(),
                "state": "running",
                "plan_id": "test-plan",
                "created_at": time.time() - 10,
                "started_at": time.time() - 9,
                "finished_at": None,
                "progress": 0.5,
                "phase": "denoise",
                "estimated_total_seconds": 20,
                "output_paths": [],
                "command": [],
                "metrics": {},
                "error": None,
                "log_path": None,
            }
            (jobs / "stale.json").write_text(json.dumps(record))
            manager = JobManager(root)
            try:
                restored = manager.get("stale")
                self.assertEqual(restored.state, JobState.FAILED)
                self.assertEqual(restored.phase, "interrupted")
                self.assertIn("daemon stopped", restored.error)
            finally:
                manager.close()

    def test_legacy_image_job_with_audio_flag_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            jobs.mkdir(parents=True)
            legacy = {
                "id": "legacy-image",
                "request": {
                    "model": "flux", "task": "image", "prompt": "hello",
                    "inputs": [],
                    "output": {
                        "type": "image", "width": 512, "height": 512,
                        "frames": 1, "fps": 24, "audio": True,
                    },
                    "sampling": {"seed": 42, "steps": 4, "guidance": None},
                    "policy": {
                        "execution": "gpu", "profile": "quality",
                        "approximation": "exact", "persistent": False,
                        "allow_fallback": True,
                    },
                    "engine_options": {},
                },
                "state": "succeeded",
                "plan_id": "gpu",
                "created_at": time.time() - 2,
                "started_at": time.time() - 1,
                "finished_at": time.time(),
                "progress": 1.0,
                "phase": "complete",
                "estimated_total_seconds": 1,
                "output_paths": [],
                "command": [],
                "metrics": {},
                "error": None,
                "log_path": None,
            }
            (jobs / "legacy-image.json").write_text(json.dumps(legacy))
            manager = JobManager(root)
            try:
                restored = manager.get("legacy-image")
                self.assertEqual(restored.state, JobState.SUCCEEDED)
                self.assertFalse(restored.request.output.audio)
            finally:
                manager.close()
    def test_successful_job_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(Path(directory))
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            spec = CommandSpec(
                argv=["/bin/sh", "-c", "echo 'phase=test 1/1'"],
                cwd=Path(directory),
            )
            record = manager.submit(request, "test-plan", spec)
            record = manager.wait(record.id, timeout=10)
            self.assertEqual(record.state, JobState.SUCCEEDED)
            self.assertTrue((Path(directory) / "jobs" / (record.id + ".json")).is_file())
            manager.close()

    def test_running_job_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(Path(directory))
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            spec = CommandSpec(
                argv=["/bin/sh", "-c", "while true; do sleep 1; done"],
                cwd=Path(directory),
            )
            record = manager.submit(request, "test-plan", spec)
            manager.cancel(record.id)
            record = manager.wait(record.id, timeout=10)
            self.assertEqual(record.state, JobState.CANCELLED)
            manager.close()

    def test_close_cancels_and_reaps_running_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "engine.pid"
            manager = JobManager(root)
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            spec = CommandSpec(
                argv=[
                    "/bin/sh", "-c",
                    "echo $$ > engine.pid; while true; do sleep 1; done",
                ],
                cwd=root,
            )
            record = manager.submit(request, "test-plan", spec)
            deadline = time.monotonic() + 5
            while not pid_file.is_file():
                if time.monotonic() >= deadline:
                    self.fail("engine process did not start")
                time.sleep(0.01)
            pid = int(pid_file.read_text())
            manager.close()
            self.assertEqual(manager.get(record.id).state, JobState.CANCELLED)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_multiphase_progress_produces_elapsed_time_and_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = JobManager(root)
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            output = root / "output.txt"
            script = (
                "import pathlib,time; "
                "print('text encoder 1/2', flush=True); time.sleep(.15); "
                "print('text encoder 2/2', flush=True); "
                "print('denoise 1/4', flush=True); time.sleep(.15); "
                "pathlib.Path('output.txt').write_text('ok')"
            )
            spec = CommandSpec(
                argv=["/usr/bin/python3", "-c", script],
                cwd=root,
                output_paths=[output],
                metadata={
                    "expected_seconds": 1.0,
                    "progress_phases": [
                        {"name": "text encoder", "weight": 0.25},
                        {"name": "denoise", "weight": 0.75},
                    ],
                },
            )
            record = manager.submit(request, "test-plan", spec)
            deadline = time.monotonic() + 5
            running = None
            while time.monotonic() < deadline:
                candidate = manager.get(record.id).as_dict()
                if candidate["progress"] >= 0.25:
                    running = candidate
                    break
                time.sleep(0.01)
            self.assertIsNotNone(running)
            self.assertGreaterEqual(running["elapsed_seconds"], 0.0)
            self.assertIsNotNone(running["estimated_remaining_seconds"])
            record = manager.wait(record.id, timeout=5)
            finished = record.as_dict()
            self.assertEqual(finished["state"], "succeeded")
            self.assertEqual(finished["progress"], 1.0)
            self.assertIsNone(finished["estimated_remaining_seconds"])
            self.assertGreater(finished["elapsed_seconds"], 0.0)
            manager.close()

    def test_unstructured_log_line_does_not_replace_known_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = JobManager(root)
            request = GenerationRequest.from_dict({"model": "test", "prompt": "hello"})
            output = root / "output.txt"
            script = (
                "import pathlib,time; "
                "print('phase=denoise 1/4', flush=True); "
                "print('backend diagnostic', flush=True); time.sleep(.25); "
                "pathlib.Path('output.txt').write_text('ok')"
            )
            spec = CommandSpec(
                argv=["/usr/bin/python3", "-c", script],
                cwd=root,
                output_paths=[output],
                metadata={
                    "progress_phases": [{"name": "denoise", "weight": 1.0}],
                },
            )
            record = manager.submit(request, "test-plan", spec)
            deadline = time.monotonic() + 5
            observed = None
            while time.monotonic() < deadline:
                candidate = manager.get(record.id)
                if candidate.state is JobState.RUNNING and candidate.progress > 0:
                    observed = candidate.phase
                    break
                time.sleep(0.01)
            self.assertEqual(observed, "denoise")
            manager.wait(record.id, timeout=5)
            manager.close()


if __name__ == "__main__":
    unittest.main()
