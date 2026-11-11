#!/usr/bin/env python3
"""Build and exercise the production connector cache using CPU-only fixtures."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ConnectedConditioningCacheTests(unittest.TestCase):
    def test_production_cache_and_held_source_identity(self):
        compiler = subprocess.check_output(['xcrun', '--find', 'clang++'], text=True).strip()
        sdk = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()
        sources = ['tests/native/ltx_conditioning_cache_test.mm',
                   'native/runtime/streaming/source_lease.cpp',
                   'native/runtime/streaming/canonical_encoding.cpp',
                   'native/runtime/memory_manifest.cpp', 'native/core/common.cpp', 'native/core/json_keys.cpp']
        with tempfile.TemporaryDirectory(prefix='tc-ltx-connected-cache-') as directory:
            binary = Path(directory) / 'cache-test'
            subprocess.run([compiler, '-std=c++20', '-O2', '-fobjc-arc', '-Wno-deprecated-declarations',
                            '-isysroot', sdk,
                            '-I', str(ROOT / 'native/platform/apple'),
                            '-I', str(ROOT / 'native/runtime/streaming'),
                            *(str(ROOT / path) for path in sources), '-framework', 'Foundation', '-o', str(binary)],
                           check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS connected-conditioning cache:', result.stdout)
            print(result.stdout, end='')


if __name__ == '__main__':
    unittest.main()
