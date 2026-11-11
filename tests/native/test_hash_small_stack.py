"""File provenance hashing must fit App/libdispatch worker thread stacks."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WorkerStackTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'darwin', 'CommonCrypto model helpers require macOS')
    def test_model_provenance_hashes_on_small_worker_stack(self):
        """Compile the actual model helpers and exercise multi-chunk input."""
        with tempfile.TemporaryDirectory(prefix='tc-model-hash-test-') as folder:
            directory = Path(folder)
            fixture = directory / 'weights'
            payload = bytes(range(256)) * 16385
            fixture.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            for model in ('ltx', 'h3', 'wan'):
                with self.subTest(model=model):
                    source = (ROOT / f'native/platform/apple/{model}_session.mm').read_text()
                    start = source.index('static std::string sha256_file(')
                    end = source.index('\n}\n', start) + 3
                    helper = source[start:end]
                    test = directory / f'{model}.cpp'
                    test.write_text('''
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <pthread.h>
''' + helper + '''
struct HashCase { std::filesystem::path file; std::string digest; };
static void *run(void *opaque) {
    auto &test = *static_cast<HashCase *>(opaque);
    test.digest = sha256_file(test.file);
    return nullptr;
}
int main(int argc, char **argv) {
    if (argc != 3) return 1;
    HashCase test{argv[1], {}};
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) return 2;
    if (pthread_attr_setstacksize(&attr, 256 * 1024)) return 3;
    pthread_t thread;
    int status = pthread_create(&thread, &attr, run, &test);
    pthread_attr_destroy(&attr);
    if (!status) status = pthread_join(thread, nullptr);
    if (status || test.digest != argv[2]) return 4;
    if (!sha256_file(test.file.string() + ".missing").empty()) return 5;
    return 0;
}
''')
                    binary = directory / model
                    subprocess.run(['clang++', '-std=c++20', '-O2',
                                    '-Wno-deprecated-declarations', str(test), '-o', str(binary)],
                                   check=True, capture_output=True)
                    result = subprocess.run([str(binary), str(fixture), expected],
                                            capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless((ROOT / 'build/native/libturbocider.dylib').is_file(), 'build native first')
    def test_sha256_on_small_worker_stack(self):
        with tempfile.TemporaryDirectory(prefix='tc-hash-test-') as folder:
            binary = Path(folder) / 'hash-worker'
            subprocess.run([
                'clang++', '-std=c++20', '-I', str(ROOT / 'native'),
                '-I', str(ROOT / 'native/core'),
                str(ROOT / 'tests/native/hash_small_stack_test.cpp'),
                '-L' + str(ROOT / 'build/native'), '-lturbocider',
                '-Wl,-rpath,' + str(ROOT / 'build/native'), '-o', str(binary),
            ], check=True, capture_output=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS: SHA-256 on a 256 KiB worker stack', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
