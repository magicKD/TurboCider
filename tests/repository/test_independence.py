"""Shipping code must not implicitly discover or import sibling projects."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
class IndependenceTests(unittest.TestCase):
    def test_no_fastmetal_shipping_aliases(self):
        for folder in ['apps', 'native', 'bindings', 'services']:
            for path in (ROOT/folder).rglob('*'):
                if path.suffix not in ['.cpp', '.mm', '.h', '.hpp', '.swift', '.c', '.m']:
                    continue
                source = path.read_text()
                # This is the real upstream checkpoint name, not a runtime alias.
                source = source.replace('FastVideo/FastMetal-1.3B-QAD', '')
                self.assertNotIn('fastmetal', source.lower(), str(path))

    def test_wan_shipping_path_is_native(self):
        session = (ROOT/'native/platform/apple/wan_session.mm').read_text()
        for token in ['posix_spawn', 'PersistentWorker', 'NSTask', 'getenv(',
                      'engine_root', 'bridge_dir']:
            self.assertNotIn(token, session)
        self.assertIn('wan::Pipeline', session)
        self.assertIn('vae/taew2_1.safetensors', session)
        self.assertFalse((ROOT/'native/platform/apple/fastmetal_session.mm').exists())
        self.assertFalse((ROOT/'tools/native/fastmetal_worker.py').exists())
        self.assertTrue((ROOT/'tools/validation/wan/python_worker.py').is_file())
        for path in ['tools/native/build.sh', 'tools/native/package.sh']:
            self.assertNotIn('fastmetal_worker', (ROOT/path).read_text())
    def test_no_sd_cpp_in_shipping_targets(self):
        forbidden = ['sd-server', 'sd-cli', 'sd_cpp', 'stable-diffusion-cpp',
                     'stable_diffusion_cpp', 'TURBOCIDER_Z_GGUF_NATIVE_GPU',
                     'TURBOCIDER_Z_IMAGE_NATIVE_ROOT']
        files = [ROOT/'tools/native/build.sh', ROOT/'tools/native/package.sh']
        for folder in ['apps', 'native', 'services', 'bindings']:
            files.extend(p for p in (ROOT/folder).rglob('*')
                         if p.suffix in ['.c', '.m', '.cpp', '.mm', '.hpp', '.h', '.swift'])
        for path in files:
            for token in forbidden:
                self.assertNotIn(token, path.read_text(), str(path))
        self.assertTrue((ROOT/'tools/validation/sd_cpp/compare_native.py').is_file())
        self.assertFalse((ROOT/'native/platform/apple/sd_cpp_session.mm').exists())
    def test_no_sibling_runtime_paths(self):
        # Historical schema/provenance names may mention an upstream project;
        # shipping code must not contain paths that discover or import it.
        forbidden=['../mac_local_ai','../mac_image_generation','../h3.c-fork',
                   '../ltx-mac','mflux/.venv','.deps/coreml']
        for folder in ['apps','native','services','bindings']:
            for p in (ROOT/folder).rglob('*'):
                if p.suffix not in ['.swift','.cpp','.mm','.hpp','.h']:continue
                for token in forbidden:self.assertNotIn(token,p.read_text(),str(p))
        llada=(ROOT/'native/platform/apple/llada_session.mm').read_text()
        self.assertNotIn('references/LLaDA-Image',llada)
        self.assertNotIn('repository / "Python/bin/python3"',llada)
        self.assertNotIn('TURBOCIDER_LLADA_WORKER',llada)
        self.assertNotIn('TURBOCIDER_LLADA_PYTHON',llada)
        self.assertNotIn('TURBOCIDER_LLADA_SOURCE',llada)
        for token in ['getenv(', 'posix_spawn', 'LLaDAWorker', 'NSTask']:
            self.assertNotIn(token, llada)
        self.assertTrue((ROOT/'tools/validation/llada/python_worker.py').is_file())
        self.assertFalse((ROOT/'tools/native/llada_worker.py').exists())
        self.assertIn('create_llada_image_native',llada)
        package=(ROOT/'tools/native/package.sh').read_text()
        self.assertNotIn('llada_worker.py',package)
    def test_managed_build_and_export(self):
        self.assertIn('dependencies.sh',(ROOT/'tools/native/build.sh').read_text())
        self.assertIn('dependencies.sh',(ROOT/'tools/native/package.sh').read_text())
        resources = (ROOT/'native/backends/coreml_resources.mm').read_text()
        for token in ['NSTask', 'launchAndReturnError', 'PYTHONPATH', 'python3']:
            self.assertNotIn(token, resources)
        self.assertIn('Core ML export is offline-only', resources)
        for path in ['tools/native/build.sh', 'tools/native/package.sh']:
            self.assertNotIn('cp tools/coreml/export_', (ROOT/path).read_text())
        self.assertNotIn('research.scripts',(ROOT/'tools/coreml/export_flux2.py').read_text())
        for path in ['tools/dependencies/build.lock.txt','tools/dependencies/coreml.lock.txt','tools/setup_dependencies.py']:
            self.assertTrue((ROOT/path).is_file())
    def test_lora_tools_are_repository_local(self):
        package = (ROOT/'tools/native/package.sh').read_text()
        self.assertNotIn('lora_runtime_cache.py', package)
        self.assertNotIn('lora_cache.mm', (ROOT/'tools/native/build.sh').read_text())
        cache=(ROOT/'tools/native/lora_runtime_cache.py').read_text()
        prepare=(ROOT/'tools/native/prepare_lora.py').read_text()
        cli=(ROOT/'apps/cli/main.mm').read_text()
        for token in ['TURBOCIDER_PREPARE_PYTHON', 'Python/bin',
                      'lora_prepare_script']:
            self.assertNotIn(token, cli)
        for name in ['prepare_lora.py', 'merge_h3_lora.py',
                     'merge_ltx_refiner.py', 'export_flux2.py', 'export_z_image.py']:
            self.assertNotIn(name, package)
        self.assertFalse((ROOT/'native/platform/apple/lora_cache.mm').exists())
        self.assertNotIn('parents[3] / "h3.c"',cache)
        self.assertNotIn('TURBOCIDER_WORKSPACE',cache)
        self.assertNotIn('/ "h3.c" / "tools"',prepare)
        self.assertNotIn('TURBOCIDER_WORKSPACE',cli)
        for path in ['native/platform/apple/h3_session.mm',
                     'native/platform/apple/ltx_session.mm']:
            source = (ROOT/path).read_text()
            for token in ['ensure_runtime_lora_cache', 'NSTask',
                          'TURBOCIDER_H3_LORA_BASE', 'TURBOCIDER_LTX_LORA_BASE']:
                self.assertNotIn(token, source, path)
        for name in ['merge_h3_lora.py','merge_ltx_refiner.py']:
            self.assertTrue((ROOT/'tools/native'/name).is_file(),name)
if __name__=='__main__':unittest.main(verbosity=2)
