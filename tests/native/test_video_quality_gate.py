#!/usr/bin/env python3
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "turbocider_video_quality_gate", ROOT / "tools/native/video_quality_gate.py"
)
QUALITY_GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUALITY_GATE)


class VideoQualityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which("ffmpeg")
        cls.ffprobe = shutil.which("ffprobe")
        if not cls.ffmpeg or not cls.ffprobe:
            raise unittest.SkipTest("ffmpeg and ffprobe are required")

    def make_video(self, directory: Path, name: str, color_a: str, color_b: str,
                   size: str = "8x8") -> Path:
        path = directory / name
        subprocess.run([
            self.ffmpeg, "-v", "error", "-y", "-f", "lavfi",
            "-i", f"color=c={color_a}:s={size}:r=2:d=1",
            "-f", "lavfi", "-i", f"color=c={color_b}:s={size}:r=2:d=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ], check=True, capture_output=True)
        return path

    def test_identical_videos_pass(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            reference = self.make_video(directory, "reference.mp4", "red", "blue")
            candidate = directory / "candidate.mp4"
            candidate.write_bytes(reference.read_bytes())
            metrics = QUALITY_GATE.compare(reference, candidate,
                                           ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
            self.assertTrue(QUALITY_GATE.passes(metrics))
            self.assertEqual(metrics["frame_count_compared"], 4)

    def test_changed_video_fails_motion_or_rgb_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            reference = self.make_video(directory, "reference.mp4", "red", "blue")
            candidate = self.make_video(directory, "candidate.mp4", "green", "green")
            metrics = QUALITY_GATE.compare(reference, candidate,
                                           ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
            self.assertFalse(QUALITY_GATE.passes(metrics))
            self.assertGreater(metrics["mean_mae_255"], 5.0)

    def test_shape_mismatch_fails_without_pairing_frames(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            reference = self.make_video(directory, "reference.mp4", "red", "blue")
            candidate = self.make_video(directory, "candidate.mp4", "red", "blue", "16x8")
            metrics = QUALITY_GATE.compare(reference, candidate,
                                           ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
            self.assertFalse(QUALITY_GATE.passes(metrics))
            self.assertFalse(metrics["shape_equal"])
            self.assertEqual(metrics["frame_count_compared"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
