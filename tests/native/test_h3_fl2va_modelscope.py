import importlib.util
import fcntl
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/h3/download_h3_fl2va_modelscope.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_h3_fl2va_modelscope", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class H3FL2VAModelScopeTests(unittest.TestCase):
    def test_source_is_pinned_modelscope_only(self):
        module = load_module()
        self.assertEqual(module.MODEL_ID, "MiniMax/MiniMax-H3")
        self.assertEqual(module.ENDPOINT, "https://modelscope.cn")
        self.assertEqual(module.REVISION, "master")
        self.assertEqual(module.PREFIX, "FL2VA/transformer")
        self.assertNotIn("huggingface", SCRIPT.read_text().lower())

    def test_exact_transformer_payload_is_pinned(self):
        module = load_module()
        shards = sorted(path for path in module.FILES if path.endswith(".safetensors"))
        self.assertEqual(len(shards), 13)
        self.assertEqual(
            shards,
            [f"FL2VA/transformer/model-{index:05d}-of-00013.safetensors"
             for index in range(1, 14)],
        )
        for record in module.FILES.values():
            self.assertGreater(record["bytes"], 0)
            self.assertEqual(len(record["sha256"]), 64)
            int(record["sha256"], 16)
        payload = sum(record["bytes"] for record in module.FILES.values())
        self.assertGreater(payload, 60 * 2**30)
        self.assertLess(payload, 63 * 2**30)

    def test_index_must_reference_exactly_the_pinned_shards(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = [Path(path).name for path in module.FILES
                      if path.endswith(".safetensors")]
            weight_map = {f"tensor.{index}": shard
                          for index, shard in enumerate(shards * 41)}
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": weight_map,
            }))
            module.validate_index(root)
            weight_map["bad"] = "model-99999-of-99999.safetensors"
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": weight_map,
            }))
            with self.assertRaisesRegex(RuntimeError, "pinned 13 shards"):
                module.validate_index(root)

    def test_manifest_rejects_a_different_base(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = {
                "schema": "turbocider-modelscope-h3-fl2va-v1",
                "repository": module.MODEL_ID,
                "revision": module.REVISION,
                "files": module.FILES,
            }
            (root / "modelscope_download.json").write_text(json.dumps(good))
            module.validate_manifest(root)
            good["repository"] = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
            (root / "modelscope_download.json").write_text(json.dumps(good))
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                module.validate_manifest(root)

    def test_download_lock_rejects_a_second_writer(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / ".modelscope_download.lock"
            lock_path.touch()
            with lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "another FL2VA"):
                    with module.download_lock(root):
                        pass


if __name__ == "__main__":
    unittest.main()
