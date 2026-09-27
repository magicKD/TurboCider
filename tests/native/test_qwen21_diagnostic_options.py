import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Qwen21DiagnosticOptionsTests(unittest.TestCase):
    def test_call_accounting_and_parsing(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen21-options-") as directory:
            binary = Path(directory) / "options-test"
            subprocess.run([
                "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "tests/native/qwen21_diagnostic_options_test.cpp"),
                "-o", str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
