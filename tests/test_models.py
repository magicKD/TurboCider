from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from turbocider.errors import ValidationError
from turbocider.models import GenerationRequest


class GenerationRequestTests(unittest.TestCase):
    def test_image_defaults_to_one_frame_without_audio(self):
        request = GenerationRequest.from_dict({
            "model": "image-model", "task": "image", "prompt": "hello"
        })
        self.assertEqual(request.output.type, "image")
        self.assertEqual(request.output.frames, 1)
        self.assertFalse(request.output.audio)

    def test_output_type_must_match_task(self):
        with self.assertRaisesRegex(ValidationError, "output.type must match"):
            GenerationRequest.from_dict({
                "model": "image-model", "task": "image", "prompt": "hello",
                "output": {"type": "video", "frames": 1},
            })

    def test_image_rejects_multiple_frames(self):
        with self.assertRaisesRegex(ValidationError, "frames=1"):
            GenerationRequest.from_dict({
                "model": "image-model", "task": "image", "prompt": "hello",
                "output": {"frames": 2},
            })

    def test_prompt_can_come_from_multimodal_input(self):
        request = GenerationRequest.from_dict({
            "model": "test",
            "inputs": [{"type": "text", "role": "prompt", "text": "hello"}],
        })
        self.assertEqual(request.prompt, "hello")
        self.assertEqual(request.output.width, 512)

    def test_invalid_dimensions_fail(self):
        with self.assertRaises(ValidationError):
            GenerationRequest.from_dict({
                "model": "test",
                "prompt": "hello",
                "output": {"width": 0, "height": 512},
            })

    def test_generation_mode_is_inferred_from_image_roles(self):
        text = GenerationRequest.from_dict({
            "model": "flux", "prompt": "hello", "task": "image",
            "output": {"type": "image"},
        })
        init = GenerationRequest.from_dict({
            "model": "flux", "prompt": "hello", "task": "image",
            "inputs": [{
                "type": "image", "role": "init_image", "path": "/tmp/init.png",
            }],
            "output": {"type": "image"},
        })
        edit = GenerationRequest.from_dict({
            "model": "flux", "prompt": "hello", "task": "image",
            "inputs": [{
                "type": "image", "role": "reference", "path": "/tmp/ref.png",
            }],
            "output": {"type": "image"},
        })
        first_frame = GenerationRequest.from_dict({
            "model": "ltx", "prompt": "hello", "task": "video",
            "inputs": [{
                "type": "image", "role": "first_frame", "path": "/tmp/first.png",
                "frame_index": 0,
            }],
        })
        keyframes = GenerationRequest.from_dict({
            "model": "ltx", "prompt": "hello", "task": "video",
            "inputs": [{
                "type": "image", "role": "last_frame", "path": "/tmp/last.png",
                "frame_index": 96,
            }],
        })
        self.assertEqual(text.resolved_mode, "text_to_image")
        self.assertEqual(init.resolved_mode, "image_to_image")
        self.assertEqual(edit.resolved_mode, "image_edit")
        self.assertEqual(first_frame.resolved_mode, "image_to_video")
        self.assertEqual(keyframes.resolved_mode, "keyframe_interpolation")

    def test_image_strength_and_frame_index_are_validated(self):
        base = {
            "model": "ltx", "prompt": "hello",
            "inputs": [{
                "type": "image", "role": "first_frame", "path": "/tmp/first.png",
            }],
        }
        for invalid in (-0.01, 1.01, math.nan):
            payload = dict(base)
            payload["inputs"] = [dict(base["inputs"][0], strength=invalid)]
            with self.assertRaisesRegex(ValidationError, "strength"):
                GenerationRequest.from_dict(payload)
        payload = dict(base)
        payload["inputs"] = [dict(base["inputs"][0], frame_index=-1)]
        with self.assertRaisesRegex(ValidationError, "frame_index"):
            GenerationRequest.from_dict(payload)

    def test_relative_media_paths_are_resolved_at_request_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            relative = Path(temporary).name + "/image.png"
            request = GenerationRequest.from_dict({
                "model": "flux", "task": "image", "prompt": "hello",
                "inputs": [{
                    "type": "image", "role": "init_image", "path": relative,
                }],
                "output": {"type": "image"},
            })
            self.assertTrue(Path(request.inputs[0].path).is_absolute())


if __name__ == "__main__":
    unittest.main()
