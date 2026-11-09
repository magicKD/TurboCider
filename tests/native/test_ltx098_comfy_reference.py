"""Check the benchmark graph without starting ComfyUI or loading weights."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
import ltx098_comfy_reference as reference


class ReferenceGraphTests(unittest.TestCase):
    def test_exact_first_frame_video_and_sampling_scope(self):
        graph = reference.workflow(0)
        self.assertEqual(graph["1"]["inputs"]["ckpt_name"], reference.MODEL_NAME)
        self.assertEqual(graph["2"]["inputs"]["clip_name"], reference.TEXT_NAME)
        image = graph["7"]["inputs"]
        self.assertEqual((image["width"], image["height"], image["length"], image["strength"]),
                         (512, 512, 121, 1.0))
        self.assertEqual(graph["5"]["inputs"]["frame_rate"], 24.0)
        self.assertEqual(graph["10"]["inputs"]["sampler_name"], "euler")
        self.assertEqual([float(x.strip()) for x in reference.SIGMAS.split(",")],
                         [1.0, .9937, .9875, .9812, .975, .9094, .725, .4219, 0.0])
        self.assertEqual(graph["13"]["class_type"], "VAEDecodeTiled")
        self.assertEqual(reference.workflow(0, "full")["13"]["class_type"], "VAEDecode")
        diagnostic = reference.workflow(0, "full", True)
        self.assertEqual([diagnostic[str(i)]["class_type"] for i in (16, 17, 18, 19)],
                         ["SaveLatent", "SaveLatent", "SaveConditioning", "SaveImage"])
        self.assertEqual(graph["14"]["inputs"]["fps"], 24.0)
        self.assertEqual(graph["8"]["inputs"]["noise_seed"], 42)
        self.assertEqual(reference.workflow(1)["8"]["inputs"]["noise_seed"], 43)
        trace = reference.workflow(1, "full", False, True)
        self.assertEqual(trace["10"], {"class_type": "LTX098EulerTrace", "inputs": {"trace_name": "trace-1"}})
        self.assertEqual(trace["12"], graph["12"])
        custom = reference.workflow(1, "full", False, False, "different prompt", 777, .75)
        self.assertEqual(custom["3"]["inputs"]["text"], "different prompt")
        self.assertEqual(custom["8"]["inputs"]["noise_seed"], 778)
        self.assertEqual(custom["7"]["inputs"]["strength"], .75)
        noise_trace = reference.workflow(0, "full", True, True, trace_decode_noise=True)
        self.assertTrue(noise_trace["10"]["inputs"]["capture_decode_noise"])
        ffn_trace = reference.workflow(0, "full", True, True, trace_first_ffn=True)
        self.assertTrue(ffn_trace["10"]["inputs"]["capture_first_ffn"])
        ffn_blocks = reference.workflow(0, "full", False, True, trace_ffn_blocks="0,14,27")
        self.assertEqual(ffn_blocks["10"]["inputs"]["capture_ffn_blocks"], "0,14,27")


if __name__ == "__main__":
    unittest.main()
