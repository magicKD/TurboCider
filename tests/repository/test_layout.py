"""Prevent obsolete entry points and research code re-entering shipping targets."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
class LayoutTests(unittest.TestCase):
    def test_single_implementation(self):
        for name in ['src','Sources','engines','scripts','model-packs','device-profiles','pyproject.toml','Package.swift','requirements-flux2.txt']:
            self.assertFalse((ROOT/name).exists(),name)
    def test_shipping_boundary(self):
        build=(ROOT/'tools/native/build.sh').read_text()
        self.assertNotIn('experimental/',build)
        self.assertNotIn('*.mm',build)
        for p in (ROOT/'native').rglob('*'):
            if p.suffix in ['.mm','.hpp','.h']:
                self.assertNotIn('vendor/',p.read_text(),str(p))
    def test_product_entries(self):
        for name in ['apps/macos/App.swift','apps/cli/main.mm','services/turbociderd/service.mm','bindings/c/include/turbocider/turbocider.h','bindings/swift/TurboCiderNative.swift','profiles/apple-m4-pro-48gb.example.json']:
            self.assertTrue((ROOT/name).is_file(),name)
if __name__=='__main__':unittest.main(verbosity=2)
