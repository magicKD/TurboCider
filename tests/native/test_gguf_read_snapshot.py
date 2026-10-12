from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class GgufReadSnapshotTests(unittest.TestCase):
    def test_clone_and_bounded_copy_preserve_original_source_and_clean_private_inode(self):
        with tempfile.TemporaryDirectory(prefix="tc-gguf-read-snapshot-test-") as raw:
            directory = Path(raw)
            binary = directory / "snapshot-test"
            sources = ["tests/native/gguf_read_snapshot_test.cpp", "native/runtime/streaming/source_lease.cpp",
                       "native/runtime/streaming/canonical_encoding.cpp", "native/runtime/memory_manifest.cpp",
                       "native/core/common.cpp", "native/core/json_keys.cpp"]
            subprocess.run(["clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                            *(str(ROOT / name) for name in sources), "-o", str(binary)], check=True)
            subprocess.run([str(binary), str(directory)], check=True)


if __name__ == "__main__":
    unittest.main()
