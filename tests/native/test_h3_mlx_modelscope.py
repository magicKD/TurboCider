import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/h3/download_fasth3_modelscope.py"
FASTVIDEO_RUNNER = ROOT / "tools/h3/run_fastvideo_mlx_fasth3.py"


class H3ModelScopeTests(unittest.TestCase):
    def test_downloader_is_modelscope_only(self):
        source = SCRIPT.read_text()
        self.assertIn('from modelscope.hub.api import HubApi', source)
        self.assertIn('MODEL_ENDPOINT = "https://modelscope.cn"', source)
        self.assertNotIn('huggingface_hub', source)
        self.assertNotIn('hf-mirror', source)
        self.assertNotIn('snapshot_download', source)

    def test_fastvideo_runner_forces_local_offline_assets(self):
        source = FASTVIDEO_RUNNER.read_text()
        self.assertIn('HF_HUB_OFFLINE', source)
        self.assertIn('TRANSFORMERS_OFFLINE', source)
        self.assertIn('install_fastvideo_namespace()', source)
        self.assertNotIn('huggingface_hub', source)
        self.assertNotIn('snapshot_download', source)

    def test_noise_fixture_tools_are_local_and_modelscope_bound(self):
        creator = (ROOT / "tools/h3/create_fasth3_noise_fixture.py").read_text()
        runner = (ROOT / "tools/h3/run_fastvideo_fasth3_noise_e2e.py").read_text()
        self.assertIn('mx.random.split', creator)
        self.assertIn('video_noise', creator)
        self.assertIn('audio_noise', creator)
        self.assertIn('noise_fixture', runner)
        self.assertNotIn('snapshot_download', creator + runner)

    def test_exact_model_id_and_component_selection(self):
        spec = importlib.util.spec_from_file_location("download_fasth3_modelscope", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(
            module.MODEL_ID,
            "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree",
        )
        self.assertIn('parser.add_argument("--model-id"', SCRIPT.read_text())
        self.assertTrue(module.wanted("transformer/diffusion_pytorch_model-00001-of-00013.safetensors", "transformer"))
        self.assertFalse(module.wanted("text_encoder/config.json", "transformer"))
        self.assertTrue(module.wanted("tokenizer/tokenizer.json", "all"))
        self.assertTrue(module.wanted("text_encoder/model-00001-of-00014.safetensors", "runtime"))
        self.assertTrue(module.wanted("vae/diffusion_pytorch_model-00001-of-00003.safetensors", "runtime"))
        self.assertTrue(module.wanted("audio_vae/diffusion_pytorch_model.safetensors", "runtime"))
        self.assertTrue(module.wanted("modular_model_index.json", "runtime"))
        self.assertFalse(module.wanted("transformer/config.json", "runtime"))


if __name__ == "__main__":
    unittest.main()
