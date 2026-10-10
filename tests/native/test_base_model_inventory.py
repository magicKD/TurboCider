import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("base_inventory", ROOT / "tools/validation/base_model_inventory.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class BaseInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tc-base-inventory-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous = inventory.ROOT
        inventory.ROOT = self.root
        self.addCleanup(setattr, inventory, "ROOT", previous)

    def test_header_only_availability_not_integrity_or_runtime_qualification(self):
        path = self.root / "weights.safetensors"
        header = json.dumps({"weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"data")
        row = inventory.inspect_weight(path)
        self.assertEqual(row["tensor_count"], 1)
        self.assertEqual(row["dtypes"], {"F16": 1})
        self.assertIs(row["content_verified"], False)

    def test_truncated_header_and_bad_range_rejected(self):
        path = self.root / "weights.safetensors"
        for raw in (b"tiny", struct.pack("<Q", 100) + b"{}"):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                inventory.inspect_weight(path)
        header = json.dumps({"weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 5]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"data")
        with self.assertRaises(ValueError):
            inventory.inspect_weight(path)

    def test_gguf_magic_and_version_required(self):
        path = self.root / "model.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<I", 3))
        self.assertEqual(inventory.inspect_weight(path)["format"], "gguf-v3")
        for prefix in (b"GGUF" + struct.pack("<I", 2), b"FAKE" + struct.pack("<I", 3)):
            path.write_bytes(prefix)
            with self.assertRaises(ValueError):
                inventory.inspect_weight(path)

    def test_empty_model_tree_cannot_claim_assets_present(self):
        rows = inventory.inventory()["rows"]
        self.assertTrue(rows)
        self.assertTrue(all(not row["assets_present"] and row["missing_components"] for row in rows))
        self.assertTrue(all(row["runtime_support_verified"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
