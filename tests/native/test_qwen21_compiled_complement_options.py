"""Host-only bounded complete-request explicit-W GPU complement admission."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class CompiledComplementOptionsTests(unittest.TestCase):
    def test_request_domain_and_source_bound_sticky_state(self):
        with tempfile.TemporaryDirectory(prefix='tc-compiled-complement-options-') as directory:
            binary=Path(directory)/'test'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'native/core'),
                str(ROOT/'tests/native/qwen21_compiled_complement_options_test.cpp'),'-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)

    def test_promoted_body_preserves_actual_component_arithmetic(self):
        original=(ROOT/'tools/profiling/qwen_base/compiled_complement.hpp').read_text()
        native=(ROOT/'native/models/qwen21/compiled_complement.hpp').read_text()
        self.assertEqual(original[original.index('inline void admit'):original.rfind('\n}')],
                         native[native.index('inline void admit'):native.rfind('\n}')])


if __name__=='__main__':unittest.main()
