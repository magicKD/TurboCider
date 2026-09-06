"""Shipping code must not implicitly discover or import sibling projects."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
class IndependenceTests(unittest.TestCase):
    def test_no_sibling_runtime_paths(self):
        forbidden=['mac_local_ai','mac_image_generation','h3.c-fork','ltx-mac','mflux/.venv','.deps/coreml']
        for folder in ['apps','native','services','bindings']:
            for p in (ROOT/folder).rglob('*'):
                if p.suffix not in ['.swift','.cpp','.mm','.hpp','.h']:continue
                for token in forbidden:self.assertNotIn(token,p.read_text(),str(p))
    def test_managed_build_and_export(self):
        self.assertIn('dependencies.sh',(ROOT/'tools/native/build.sh').read_text())
        self.assertIn('dependencies.sh',(ROOT/'tools/native/package.sh').read_text())
        self.assertIn('toolchains/coreml/bin/python3',(ROOT/'native/backends/coreml_resources.mm').read_text())
        self.assertNotIn('research.scripts',(ROOT/'tools/coreml/export_flux2.py').read_text())
        for path in ['tools/dependencies/build.lock.txt','tools/dependencies/coreml.lock.txt','tools/setup_dependencies.py']:
            self.assertTrue((ROOT/path).is_file())
if __name__=='__main__':unittest.main(verbosity=2)
