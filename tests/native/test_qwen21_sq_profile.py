import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sq_profile', ROOT / 'tools/native/embed_qwen21_sq_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

class FrozenProfileTests(unittest.TestCase):
    def test_frozen_profile_embeds_without_external_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / 'embedded.hpp'
            profile.generate(profile.ASSETS, output)
            source = output.read_text()
            self.assertIn('inline constexpr uint16_t inverse_bits[]', source)
            self.assertIn('777c4179010b3f662515013bbd51e1e57e792b9635b3d55a9fea073956a9eedc', source)
            self.assertNotIn('/Users/', source)

    def test_corrupt_payload_and_index_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            shutil.copytree(profile.ASSETS, directory / 'asset')
            asset = directory / 'asset'
            payload = asset / 'sq_convrot_v1.f16le'
            original = payload.read_bytes()
            payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                profile.generate(asset, directory / 'out.hpp')
            payload.write_bytes(original)
            index = asset / 'sq_convrot_v1.json'
            data = json.loads(index.read_text())
            data['entries'][0]['offset'] = 1
            index.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                profile.generate(asset, directory / 'out.hpp')

    def test_runtime_identity_binds_payload_and_mapping(self):
        spec = importlib.util.spec_from_file_location('identity', ROOT / 'tools/native/generate_runtime_build_identity.py')
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        paths = {entry['path'] for entry in identity.source_inputs(ROOT)}
        for name in ('sq_convrot_v1.json', 'sq_convrot_v1.f16le'):
            self.assertIn('native/models/qwen21/calibration/' + name, paths)

if __name__ == '__main__':
    unittest.main()
