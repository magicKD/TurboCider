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
        self.assertIn('SOURCES=(',build)
        self.assertNotIn('find native',build)
        for p in (ROOT/'native').rglob('*'):
            if p.suffix in ['.cpp','.mm','.hpp','.h']:
                self.assertNotIn('vendor/',p.read_text(),str(p))
    def test_product_entries(self):
        for name in ['apps/macos/App.swift','apps/cli/main.mm','services/turbociderd/service.mm','bindings/c/include/turbocider/turbocider.h','bindings/swift/TurboCiderNative.swift','profiles/apple-m4-pro-48gb.example.json']:
            self.assertTrue((ROOT/name).is_file(),name)
    def test_package_contains_video_runtime_resources(self):
        build=(ROOT/'tools/native/build.sh').read_text()
        self.assertIn('TURBOCIDER_DEPLOYMENT_TARGET',build)
        self.assertNotIn('mmacosx-version-min=15.0',build)
        self.assertNotIn('apple-macosx15.0',build)
        package=(ROOT/'tools/native/package.sh').read_text()
        for name in ['h3_shaders.metal','ltx_shaders.metal',
                     'ltx-video-finalizer','ltx-video-vae-decode',
                     'lora_runtime_cache.py','merge_h3_lora.py',
                     'merge_ltx_refiner.py','fastmetal_worker.py']:
            self.assertIn(name,package)
        self.assertIn('MLX_LICENSE_PATH',package)
        self.assertNotIn('mlx-0.32.0.dist-info',package)
        self.assertIn('rm -rf "$APP" "$ROOT/dist/cli"',package)
        self.assertIn('<<PLIST',package)
        session=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn('dladdr((void*)&ltx_runtime_resource',session)
        self.assertIn('TURBOCIDER_RESOURCE_DIR',session)
if __name__=='__main__':unittest.main(verbosity=2)
