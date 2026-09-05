from __future__ import annotations

import unittest
from pathlib import Path

from turbocider.errors import PlanUnavailableError
from turbocider.models import (
    ExecutionPlan,
    GenerationRequest,
    ModelDescriptor,
)
from turbocider.policy import choose_plan


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelDescriptor(
            id="test",
            name="Test",
            engine="h3",
            version="1",
            capabilities={},
            config={"path": str(Path(__file__).resolve())},
            plans=[
                ExecutionPlan.from_dict({
                    "id": "gpu",
                    "execution": "gpu",
                    "quality": "exact",
                    "profile": "*",
                    "production": True,
                    "priority": 100,
                    "requirements": {"config_paths": ["path"]},
                }),
                ExecutionPlan.from_dict({
                    "id": "hybrid",
                    "execution": "gpu_ane",
                    "quality": "validated",
                    "profile": "quality",
                    "production": True,
                    "priority": 130,
                    "requirements": {"persistent": True},
                }),
            ],
        )

    def request(self, **policy):
        return GenerationRequest.from_dict({
            "model": "test",
            "prompt": "hello",
            "policy": policy,
        })

    def test_auto_uses_gpu_when_hybrid_not_compatible(self):
        self.assertEqual(choose_plan(self.model, self.request()).id, "gpu")

    def test_auto_uses_ranked_hybrid_when_persistent(self):
        request = self.request(persistent=True)
        self.assertEqual(choose_plan(self.model, request).id, "hybrid")

    def test_explicit_hybrid_fails_closed(self):
        with self.assertRaises(PlanUnavailableError):
            choose_plan(self.model, self.request(execution="gpu_ane"))

    def test_auto_rejects_unvalidated_device_but_explicit_hybrid_forces_it(self):
        hybrid = ExecutionPlan.from_dict({
            "id": "device-hybrid",
            "execution": "gpu_ane",
            "quality": "validated",
            "profile": "quality",
            "production": True,
            "priority": 200,
            "requirements": {
                "persistent": True,
                "device_profiles": ["profile-that-is-not-installed"],
            },
        })
        model = ModelDescriptor(
            id=self.model.id,
            name=self.model.name,
            engine=self.model.engine,
            version=self.model.version,
            capabilities=self.model.capabilities,
            config=self.model.config,
            plans=[self.model.plans[0], hybrid],
        )
        self.assertEqual(
            choose_plan(model, self.request(persistent=True)).id,
            "gpu",
        )
        self.assertEqual(
            choose_plan(
                model,
                self.request(execution="gpu_ane", persistent=True),
            ).id,
            "device-hybrid",
        )

    def test_shape_requirement_can_gate_fps(self):
        hybrid = ExecutionPlan.from_dict({
            "id": "fixed-fps",
            "execution": "gpu_ane",
            "quality": "validated",
            "profile": "quality",
            "production": True,
            "priority": 200,
            "requirements": {
                "shapes": [{
                    "width": 512, "height": 512, "frames": 22, "fps": 16
                }]
            },
        })
        model = ModelDescriptor(
            id=self.model.id,
            name=self.model.name,
            engine=self.model.engine,
            version=self.model.version,
            capabilities=self.model.capabilities,
            config=self.model.config,
            plans=[self.model.plans[0], hybrid],
        )
        self.assertEqual(choose_plan(model, self.request()).id, "gpu")
        matching = GenerationRequest.from_dict({
            "model": "test", "prompt": "hello",
            "output": {"width": 512, "height": 512, "frames": 22, "fps": 16},
        })
        self.assertEqual(choose_plan(model, matching).id, "fixed-fps")

    def test_mode_requirement_routes_image_edit_away_from_hybrid(self):
        gpu = ExecutionPlan.from_dict({
            "id": "gpu-edit", "execution": "gpu", "quality": "exact",
            "priority": 100, "production": True,
            "requirements": {"modes": ["image_edit"]},
        })
        hybrid = ExecutionPlan.from_dict({
            "id": "hybrid-standard", "execution": "gpu_ane",
            "quality": "validated", "priority": 200, "production": True,
            "requirements": {
                "persistent": True,
                "modes": ["text_to_image", "image_to_image"],
            },
        })
        model = ModelDescriptor(
            id="flux", name="Flux", engine="flux2", version="1",
            capabilities={}, config={}, plans=[gpu, hybrid],
        )
        request = GenerationRequest.from_dict({
            "model": "flux", "task": "image", "prompt": "edit",
            "mode": "image_edit",
            "inputs": [{
                "type": "image", "role": "reference", "path": "/tmp/ref.png",
            }],
            "output": {"type": "image"},
            "policy": {"persistent": True},
        })
        self.assertEqual(choose_plan(model, request).id, "gpu-edit")
        request = GenerationRequest.from_dict({
            "model": "flux", "task": "image", "prompt": "edit",
            "mode": "image_edit",
            "inputs": [{
                "type": "image", "role": "reference", "path": "/tmp/ref.png",
            }],
            "output": {"type": "image"},
            "policy": {"execution": "gpu_ane", "persistent": True},
        })
        with self.assertRaises(PlanUnavailableError):
            choose_plan(model, request)


if __name__ == "__main__":
    unittest.main()
