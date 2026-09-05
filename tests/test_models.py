from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
