from __future__ import annotations

import json
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from turbocider.workers.ltx_pipeline import (
    _bf16_lookup,
    audio_mux_command,
    prepare_i2v_latents,
    preprocess_image_bf16,
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

    @patch("turbocider.workers.ltx_pipeline.capture")
    @patch("turbocider.workers.ltx_pipeline.run")
    def test_image_preprocessing_writes_planar_bf16(self, run, capture):
        raw = bytes((index % 256 for index in range(32 * 32 * 3)))
        capture.return_value = raw
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pixels.bf16"
            preprocess_image_bf16(
                root / "source.png", output,
                width=32, height=32, crf=33, cwd=root,
            )
            values = struct.unpack("<3072H", output.read_bytes())

        lookup = _bf16_lookup()
        pixels = 32 * 32
        self.assertEqual(values[0], lookup[raw[0]])
        self.assertEqual(values[1], lookup[raw[3]])
        self.assertEqual(values[pixels], lookup[raw[1]])
        self.assertEqual(values[2 * pixels], lookup[raw[2]])
        self.assertEqual(run.call_count, 1)
        filter_value = capture.call_args.args[0][
            capture.call_args.args[0].index("-vf") + 1
        ]
        self.assertIn("force_original_aspect_ratio=increase", filter_value)
        self.assertIn("flags=lanczos", filter_value)
        self.assertIn("crop=32:32", filter_value)

    @patch("turbocider.workers.ltx_pipeline.preprocess_image_bf16")
    @patch("turbocider.workers.ltx_pipeline.run")
    def test_prepare_i2v_encodes_both_native_stages(self, run, preprocess):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            first_frame = root / "first.png"
            helper = root / "bench_mlx_video_vae"
            video_vae = root / "video-vae.safetensors"
            for path in (first_frame, helper, video_vae):
                path.touch()

            def create_latent(command, **_kwargs):
                height = int(command[4])
                width = int(command[5])
                Path(command[7]).write_bytes(
                    b"\0" * ((height // 32) * (width // 32) * 128 * 2)
                )

            run.side_effect = create_latent
            args = Namespace(
                first_frame=first_frame,
                image_strength=0.7,
                image_crf=31,
                video_vae_helper=helper,
                ltx_root=root,
                video_vae=video_vae,
                artifact_dir=artifact_dir,
            )

            report = prepare_i2v_latents(args, 704, 448)

            self.assertEqual(preprocess.call_count, 2)
            self.assertEqual(
                (report["stages"]["stage1"]["width"],
                 report["stages"]["stage1"]["height"],
                 report["stages"]["stage1"]["latent_rows"]),
                (352, 224, 77),
            )
            self.assertEqual(
                (report["stages"]["stage2"]["width"],
                 report["stages"]["stage2"]["height"],
                 report["stages"]["stage2"]["latent_rows"]),
                (704, 448, 308),
            )
            self.assertEqual(run.call_count, 2)
            self.assertEqual(report["strength"], 0.7)

    @patch("turbocider.workers.ltx_pipeline.preprocess_image_bf16")
    @patch("turbocider.workers.ltx_pipeline.run")
    def test_prepare_i2v_rejects_missing_encoder_output(self, run, preprocess):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            first_frame = root / "first.png"
            helper = root / "bench_mlx_video_vae"
            video_vae = root / "video-vae.safetensors"
            for path in (first_frame, helper, video_vae):
                path.touch()
            args = Namespace(
                first_frame=first_frame, image_strength=1.0, image_crf=33,
                video_vae_helper=helper, ltx_root=root,
                video_vae=video_vae, artifact_dir=artifact_dir,
            )
            with self.assertRaisesRegex(RuntimeError, "did not create"):
                prepare_i2v_latents(args, 704, 448)


if __name__ == "__main__":
    unittest.main()
