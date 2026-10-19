"""Host checks for declared-rank comparison boundaries; no model loading."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
from qwen_student_rank_report import comparable_request


class StudentRankReportTests(unittest.TestCase):
    def request(self):
        return dict(model="qwen-image-2.1", operation="image.edit", prompt="red teapot",
            width=512, height=512, steps=6, seed=73, qwen21_reference_size=512,
            output="a.png", loras=[dict(path="r256.safetensors", strength=1., role="transformer")],
            inputs=[dict(kind="image", role="reference", path="source.png")])

    def test_only_declared_adapter_path_and_output_are_excluded(self):
        a = self.request()
        saved = copy.deepcopy(a)
        b = copy.deepcopy(a)
        b["output"] = "b.png"
        b["loras"][0]["path"] = "r128.safetensors"
        self.assertEqual(comparable_request(a), comparable_request(b))
        self.assertEqual(a, saved)

    def test_steps_seed_prompt_reference_and_adapter_strength_remain_controls(self):
        a = self.request()
        for key, value in (("steps", 4), ("seed", 74), ("prompt", "blue teapot"),
                           ("qwen21_reference_size", 1024), ("width", 1024)):
            b = copy.deepcopy(a)
            b[key] = value
            with self.subTest(key=key):
                self.assertNotEqual(comparable_request(a), comparable_request(b))
        for key, value in (("strength", .5), ("role", "text_encoder")):
            b = copy.deepcopy(a)
            b["loras"][0][key] = value
            self.assertNotEqual(comparable_request(a), comparable_request(b))
        b = copy.deepcopy(a)
        b["inputs"][0]["path"] = "different.png"
        self.assertNotEqual(comparable_request(a), comparable_request(b))


if __name__ == "__main__":
    unittest.main()
