import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/h3/download_h3_vae_modelscope.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_h3_vae_modelscope", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class H3VaeModelScopeTests(unittest.TestCase):
    def test_source_and_payload_are_modelscope_only(self):
        module = load_module()
        self.assertEqual(module.MODEL_ID, "MiniMax/MiniMax-H3")
        self.assertEqual(module.ENDPOINT, "https://modelscope.cn")
        self.assertEqual(module.REVISION, "master")
        self.assertNotIn("huggingface", SCRIPT.read_text().lower())
        self.assertEqual(
            set(module.FILES),
            {
                "FL2VA/video_vae/config.json",
                "FL2VA/video_vae/source/config.json",
                "FL2VA/video_vae/source/model.safetensors",
                "FL2VA/audio_vae/config.json",
                "FL2VA/audio_vae/metadata.json",
                "FL2VA/audio_vae/model.safetensors",
            },
        )
        payload = sum(module.FILES.values())
        self.assertGreater(payload, 10 * 2**30)
        self.assertLess(payload, 11 * 2**30)

    def test_download_lock_rejects_a_second_writer(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / ".modelscope_vae_download.lock"
            lock_path.touch()
            import fcntl
            with lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "another H3 VAE"):
                    with module.download_lock(root):
                        pass


if __name__ == "__main__":
    unittest.main()
