"""Shipping code must not implicitly discover or import sibling projects."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
class IndependenceTests(unittest.TestCase):
    def test_no_sibling_runtime_paths(self):
        # Historical schema/provenance names may mention an upstream project;
        # shipping code must not contain paths that discover or import it.
        forbidden=['../mac_local_ai','../mac_image_generation','../h3.c-fork',
                   '../ltx-mac','mflux/.venv','.deps/coreml']
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
    def test_lora_tools_are_repository_local(self):
        cache=(ROOT/'tools/native/lora_runtime_cache.py').read_text()
        prepare=(ROOT/'tools/native/prepare_lora.py').read_text()
        cli=(ROOT/'apps/cli/main.mm').read_text()
        bridge=(ROOT/'native/platform/apple/lora_cache.mm').read_text()
        self.assertNotIn('parents[3] / "h3.c"',cache)
        self.assertNotIn('TURBOCIDER_WORKSPACE',cache)
        self.assertNotIn('/ "h3.c" / "tools"',prepare)
        self.assertNotIn('TURBOCIDER_WORKSPACE',cli)
        self.assertNotIn('TURBOCIDER_WORKSPACE',bridge)
        for name in ['merge_h3_lora.py','merge_ltx_refiner.py']:
            self.assertTrue((ROOT/'tools/native'/name).is_file(),name)
if __name__=='__main__':unittest.main(verbosity=2)
