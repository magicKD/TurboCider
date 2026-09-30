"""CPU-only release identity check; optional installed r128 geometry audit."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
R128_SHA256 = 'bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b'


class Qwen21ViggleAdapterTests(unittest.TestCase):
    def test_released_rank_identity_and_unknown_filename_rejection(self):
        with tempfile.TemporaryDirectory(prefix='tc-qwen21-viggle-') as directory:
            binary = Path(directory) / 'viggle-test'
            subprocess.run([
                'clang++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                str(ROOT / 'tests/native/qwen21_viggle_adapter_test.cpp'),
                '-o', str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True)

    @unittest.skipUnless(os.environ.get('TURBOCIDER_TEST_QWEN21_R128'),
                         'set TURBOCIDER_TEST_QWEN21_R128 for the installed-asset CPU audit')
    def test_installed_adapter_has_all_227_runtime_projection_pairs(self):
        path = Path(os.environ['TURBOCIDER_TEST_QWEN21_R128'])
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), R128_SHA256)
        with path.open('rb') as stream:
            header_length = struct.unpack('<Q', stream.read(8))[0]
            self.assertLess(header_length, path.stat().st_size - 8)
            header = json.loads(stream.read(header_length))
        metadata = json.loads(header['__metadata__']['lora_adapter_metadata'])
        self.assertEqual(metadata['transformer.r'], 128)
        self.assertEqual(metadata['transformer.lora_alpha'], 128)
        self.assertFalse(metadata['transformer.use_rslora'])
        tensors = {key: value for key, value in header.items() if key != '__metadata__'}
        self.assertEqual(len(tensors), 454)
        stems = {key.removesuffix('.lora_A.weight') for key in tensors
                 if key.endswith('.lora_A.weight')}
        expected = {'transformer.modulation.1',
                    'transformer.time_text_embed.timestep_embedder.linear_1',
                    'transformer.time_text_embed.timestep_embedder.linear_2'}
        for block in range(32):
            prefix = f'transformer.transformer_blocks.{block}.'
            expected.update(prefix + target for target in (
                'attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0',
                'img_mlp.gate_layer', 'img_mlp.proj', 'img_mlp.out',
            ))
        self.assertEqual(stems, expected)
        for stem in stems:
            down = tensors[stem + '.lora_A.weight']
            up = tensors[stem + '.lora_B.weight']
            self.assertEqual(down['dtype'], 'BF16')
            self.assertEqual(up['dtype'], 'BF16')
            self.assertEqual(down['shape'][0], 128)
            self.assertEqual(up['shape'][1], 128)
            self.assertEqual(len(down['shape']), 2)
            self.assertEqual(len(up['shape']), 2)


if __name__ == '__main__':
    unittest.main()
