import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class H3MLXCacheContractTests(unittest.TestCase):
    def test_cache_is_identity_bound_and_atomic(self):
        header = (ROOT / "native/models/h3_mlx/prompt_cache.hpp").read_text()
        source = (ROOT / "native/platform/apple/h3_mlx_prompt_cache.mm").read_text()
        conditioner = (ROOT / "native/models/h3_mlx/conditioner.cpp").read_text()
        for token in [
            "prompt_cache_identity",
            "schema=h3-prompt-cache-v2",
            "Qwen3-VL:first-50-language-layers",
            "numpy-float32-cody-waite-fma-v1",
            "load_prompt_cache",
            "save_prompt_cache",
        ]:
            self.assertIn(token, header + source + conditioner)
        self.assertIn("std::filesystem::rename", source)
        self.assertIn(".tmp.", source)
        self.assertIn('identity + ".safetensors"', source)
        self.assertIn("tokenizer_root_", conditioner)

    def test_cache_has_no_python_or_objc_in_model_layer(self):
        for path in (ROOT / "native/models/h3_mlx").glob("*"):
            if path.suffix in {".cpp", ".hpp"}:
                text = path.read_text()
                self.assertNotIn("#import <Foundation", text)
                self.assertNotIn("objc/", text)


if __name__ == "__main__":
    unittest.main()
