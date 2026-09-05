from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from turbocider.workers.ltx_pipeline import (
    audio_mux_command,
    validate_video_frames,
)


class LTXWorkerTests(unittest.TestCase):
    def test_audio_mux_does_not_truncate_video(self):
        command = audio_mux_command(
            Path("video.mp4"), Path("audio.wav"), Path("muxed.mp4")
        )
        self.assertNotIn("-shortest", command)
        self.assertEqual(command[-1], "muxed.mp4")

    @patch("turbocider.workers.ltx_pipeline.subprocess.run")
    def test_final_media_must_preserve_all_frames(self, run):
        run.return_value.stdout = json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "nb_frames": "97",
                    "nb_read_frames": "97",
                }
            ]
        })
        report = validate_video_frames(Path("output.mp4"), 97)
        self.assertEqual(report["validated_video_frames"], 97)

        run.return_value.stdout = json.dumps({
            "streams": [{"codec_type": "video", "nb_read_frames": "94"}]
        })
        with self.assertRaisesRegex(RuntimeError, "94 video frames"):
            validate_video_frames(Path("output.mp4"), 97)


if __name__ == "__main__":
    unittest.main()
