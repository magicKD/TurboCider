"""Node contracts, actual CPU tensors, alpha, interruption and batch requests.

Requires torch/numpy/Pillow (the existing ComfyUI environment); never weights.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tc_nodes", ROOT / "integrations/comfyui/turbocider/nodes.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class NodeTests(unittest.TestCase):
    def backend(self):
        return {"url": "http://127.0.0.1:8765", "token": "", "installation_id": "i", "loras": [],
                "catalog_loras": [{"id": "l", "modelID": "flux", "name": "adapter"}, {"id": "q", "modelID": "qwen"}],
                "model": {"id": "flux", "operations": ["image.generate", "image.edit", "image.transform"],
                          "default_steps": 4, "default_residency": "resident", "supports_lora": True,
                          "default_lora_strategy": "in_memory_merge", "lora_strategies": ["in_memory_merge"]}}

    def png(self, alpha=128):
        image = Image.new("RGBA", (16, 16), (25, 50, 200, alpha))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def test_rgba_roundtrip_and_inverse_alpha(self):
        image, mask = nodes.decode_image(self.png())
        self.assertEqual(tuple(image.shape), (1, 16, 16, 3))
        self.assertEqual(tuple(mask.shape), (1, 16, 16))
        self.assertAlmostEqual(mask[0, 0, 0].item(), 127 / 255, places=6)
        self.assertAlmostEqual(image[0, 0, 0, 2].item(), 200 / 255, places=6)

    def test_input_batch_order_no_transpose_and_finite_validation(self):
        uploaded = []
        client = types.SimpleNamespace(upload=lambda data: uploaded.append(data) or str(len(uploaded)))
        images = torch.stack([torch.zeros(16, 24, 3), torch.ones(16, 24, 3)])
        with patch.object(nodes, "check_interrupt"):
            inputs = nodes.upload_images(client, images)
            self.assertEqual([i["asset_id"] for i in inputs], ["1", "2"])
            with Image.open(io.BytesIO(uploaded[1])) as image:
                self.assertEqual(image.size, (24, 16))
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            images[0, 0, 0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                nodes.upload_images(client, images)

    def test_lora_chain_immutable_and_family_strategy_guard(self):
        base = self.backend()
        before = copy.deepcopy(base)
        first = nodes.TurboCiderLoRA().attach(base, "l", 0.7)[0]
        second = nodes.TurboCiderLoRA().attach(first, "l", 0.2)[0]
        self.assertEqual(base, before)
        self.assertEqual(len(first["loras"]), 1)
        self.assertEqual([x["strength"] for x in second["loras"]], [0.7, 0.2])
        for key, strategy in [("q", "auto"), ("l", "inference_time")]:
            with self.assertRaises(ValueError):
                nodes.TurboCiderLoRA().attach(base, key, 1, strategy)

    def test_compose_uses_model_defaults_and_native_admission(self):
        request = nodes.compose(self.backend(), "fox", 512, 768, 0, 42, '{"guidance":3}', "image.generate")
        self.assertEqual(request["steps"], 4)
        self.assertEqual(request["guidance"], 3)
        for text in ['{"output":"/tmp/a"}', '{"seed":7}', '{"x":1,"x":2}', '{"x":NaN}', '[]']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                nodes.compose(self.backend(), "fox", 512, 512, 4, 42, text, "image.generate")
        with self.assertRaisesRegex(ValueError, "does not support"):
            nodes.compose(self.backend(), "fox", 512, 512, 4, 42, "{}", "video.generate")

    def test_batch_increment_wrap_and_transform_strength(self):
        requests = []
        client = types.SimpleNamespace(plan=lambda *a: {}, submit=lambda _, r: requests.append(copy.deepcopy(r)) or str(len(requests)), artifact=lambda _: self.png())
        with patch.object(nodes, "client_for", return_value=client), patch.object(nodes, "check_interrupt"), patch.object(nodes, "wait_job", return_value={"state": "succeeded"}), patch.object(nodes, "upload_images", return_value=[{"kind": "image", "role": "init_image", "asset_id": "a"}]):
            images, masks, _ = nodes.TurboCiderEdit().generate(self.backend(), "fox", 512, 512, 4, 2147483647, 2, images=torch.zeros(1, 16, 16, 3), operation="image.transform", strength=0.4)
        self.assertEqual([r["seed"] for r in requests], [2147483647, 0])
        self.assertEqual(requests[0]["inputs"][0]["strength"], 0.4)
        self.assertNotIn("strength", requests[0])
        self.assertEqual(images.shape[0], 2)
        self.assertEqual(masks.shape[0], 2)

    def test_interrupt_and_timeout_cancel_exact_job(self):
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = lambda _: types.SimpleNamespace(update_absolute=lambda *a: None)
        cancelled = []
        client = types.SimpleNamespace(status=lambda _: {"state": "running"}, cancel=lambda job: cancelled.append(job))
        with patch.dict(sys.modules, {"comfy.utils": utils}), patch.object(nodes, "check_interrupt", side_effect=InterruptedError("stop")):
            with self.assertRaises(InterruptedError):
                nodes.wait_job(client, "my-job", 60)
        with patch.dict(sys.modules, {"comfy.utils": utils}):
            with self.assertRaises(TimeoutError):
                nodes.wait_job(client, "timeout-job", 0)
        self.assertEqual(cancelled, ["my-job", "timeout-job"])

    def test_submit_transport_failure_is_not_retried(self):
        calls = []
        def failed(*args):
            calls.append(args)
            raise RuntimeError("ambiguous submission")
        client = types.SimpleNamespace(plan=lambda *a: {}, submit=failed)
        with patch.object(nodes, "client_for", return_value=client), patch.object(nodes, "check_interrupt"):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                nodes.TurboCiderGenerate().generate(self.backend(), "fox", 512, 512, 4, 42)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
