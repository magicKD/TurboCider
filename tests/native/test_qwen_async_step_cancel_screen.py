"""Portable cancellation receipt/PNG checks; no model or device work."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_async_step_cancel_screen as screen


class AsyncStepCancelScreenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="tc-async-cancel-host-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.native = dict(status="complete", successful_full_requests=8,
                           cancelled_requests=2, cases=[])
        for task, target, labels in (
                ("generate", 1, ("sync", "retry", "restore")),
                ("edit", 2, ("sync", "sync-hit", "retry", "retry-hit", "restore"))):
            entries = []
            for label in labels:
                filename = f"{task}-{label}.png"
                Image.new("RGBA", (512, 512), (19, 73, 127, 255)).save(self.root / filename)
                entries.append(dict(label=label, filename=filename))
            self.native["cases"].append(dict(
                case=task, cancel_after_submit=target, retained_steps_at_cancel=target,
                completed_progress_before_cancel=0, cancelled_png_absent=True,
                cancelled=True, successful=entries))

    def test_complete_retries_and_sync_restoration_compare_all_rgba(self):
        result = screen.analyze(self.root, self.native)
        self.assertEqual(result["successful_full_requests"], 8)
        self.assertEqual(result["cancelled_requests"], 2)
        self.assertEqual(result["rgba_exact_comparisons"], 6)
        self.assertIs(result["no_unchecked_completed_progress"], True)

    def test_incomplete_or_partial_success_counts_are_not_full_requests(self):
        for key, value in (("status", "running"), ("successful_full_requests", 7),
                           ("cancelled_requests", 1)):
            native = copy.deepcopy(self.native)
            native[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.analyze(self.root, native)

    def test_wrong_pending_boundary_or_premature_progress_is_rejected(self):
        for case_index in (0, 1):
            for key, value in (("cancel_after_submit", 3), ("retained_steps_at_cancel", 3),
                               ("completed_progress_before_cancel", 1),
                               ("cancelled_png_absent", False), ("cancelled", False)):
                native = copy.deepcopy(self.native)
                native["cases"][case_index][key] = value
                with self.subTest(case=case_index, key=key), self.assertRaises(ValueError):
                    screen.analyze(self.root, native)

    def test_actual_cancelled_file_overrides_an_absent_receipt(self):
        for task in ("generate", "edit"):
            path = self.root / f"{task}-must-not-export.png"
            Image.new("RGBA", (512, 512)).save(path)
            with self.subTest(task=task), self.assertRaises(ValueError):
                screen.analyze(self.root, self.native)
            path.unlink()

    def test_missing_retry_hit_or_policy_restore_is_rejected(self):
        for case_index in (0, 1):
            native = copy.deepcopy(self.native)
            native["cases"][case_index]["successful"].pop()
            with self.subTest(case=case_index), self.assertRaises(ValueError):
                screen.analyze(self.root, native)

    def test_alpha_change_and_incomplete_image_are_rejected(self):
        path = self.root / "edit-retry.png"
        for size, alpha in (((512, 512), 254), ((512, 511), 255)):
            Image.new("RGBA", size, (19, 73, 127, alpha)).save(path)
            with self.subTest(size=size, alpha=alpha), self.assertRaises(ValueError):
                screen.analyze(self.root, self.native)


if __name__ == "__main__":
    unittest.main()
