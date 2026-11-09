"""Validate the documented HunyuanVideo 1.5 step-distilled workflow."""
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[2] / "tools/validation"
sys.path.insert(0, str(TOOLS))
from hunyuan15_comfy_reference import workflow


class Hunyuan15WorkflowTests(unittest.TestCase):
    def test_eight_step_uses_step_distilled_weights(self):
        graph = workflow("fox", 42, 512, 512, 97, 24, 8)
        self.assertIn("i2v_step_distilled_fp16", graph["1"]["inputs"]["unet_name"])
        self.assertEqual(graph["11"]["inputs"]["steps"], 8)
        self.assertEqual(graph["11"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["10"]["inputs"]["shift"], 7.0)
        self.assertEqual(graph["9"]["inputs"]["start_image"], ["6", 0])
        self.assertEqual(graph["9"]["inputs"]["clip_vision_output"], ["8", 0])

    def test_tiled_decode_follows_saved_latent(self):
        graph = workflow("fox", 42, 512, 512, 97, 24, 8, save_latent=True, tiled_vae=True)
        self.assertEqual(graph["15"]["class_type"], "SaveLatent")
        self.assertEqual(graph["15"]["inputs"]["samples"], ["11", 0])
        self.assertEqual(graph["12"]["class_type"], "VAEDecodeTiled")
        self.assertEqual(graph["12"]["inputs"]["samples"], ["15", 0])
        self.assertEqual(graph["12"]["inputs"]["temporal_size"], 4096)


if __name__ == "__main__":
    unittest.main()
