"""Prevent obsolete entry points and research code re-entering shipping targets."""
from pathlib import Path
import json
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[2]
class LayoutTests(unittest.TestCase):
    def test_runtime_weight_ane_stays_opt_in(self):
        for name in ['tools/native/build.sh', 'tools/native/package.sh']:
            source = (ROOT/name).read_text()
            for experimental in ['ane-runtime-probe',
                                 'export_runtime_ane.py', 'build_ane_runtime_probe.sh']:
                self.assertNotIn(experimental, source, name)
        makefile = (ROOT/'Makefile').read_text()
        self.assertIn('build-runtime-ane-probe:', makefile)
        self.assertIn('test-runtime-ane: test-runtime-ane-host', makefile)
        default_tests = makefile.split('\ntest:\n', 1)[1].split('\n# No real model', 1)[0]
        self.assertIn('$(MAKE) test-acceleration-contract\n', default_tests)
        contracts = makefile.split('\ntest-acceleration-contract:', 1)[1].split('\ntest-runtime-ane-host:', 1)[0]
        self.assertTrue(contracts.startswith(' test-runtime-ane-host\n'))
        self.assertIn('test_runtime_ane_model_screen.py', contracts)
        self.assertIn('test_runtime_lora_shared_graph_switch.py', contracts)
        self.assertIn('test_cli_ane_override.py', contracts)
        self.assertIn('test_ane_placement.py', contracts)
        self.assertNotIn('test_ane_runtime.py', contracts)
        self.assertNotIn('$(MAKE) test-runtime-ane\n', contracts)
        self.assertNotIn('$(MAKE) test-runtime-ane\n', default_tests)
        self.assertNotIn('test_ane_runtime.py', default_tests)
        for model in ['qwen21', 'z_image']:
            source = (ROOT/f'native/models/{model}_module.cpp').read_text()
            self.assertIn('r.hybrid_mlp_mode == "runtime"', source)
            self.assertIn('r.execution == "gpu_ane"', source)
    def test_runtime_ane_sources_are_portable(self):
        # Keep artifacts, local sibling research trees and user directories
        # out of reusable execution code and offline preparation tools.
        sources = [*sorted((ROOT/'native/backends').glob('ane_*'))]
        sources += [ROOT/name for name in ['tools/coreml/export_runtime_ane.py',
                     'tools/native/build_ane_runtime_probe.sh',
                     'tools/native/build_ane_ffn_test.sh',
                     'tools/native/ane_runtime_probe.cpp',
                     'tools/validation/runtime_ane_common.py',
                     'tools/validation/qwen21_ane_placement.py',
                     'tools/validation/runtime_ane_memory.py',
                     'tools/validation/runtime_ane_model_screen.py',
                     'tools/validation/runtime_lora_shared_graph_switch.py']]
        sources += [ROOT/'native/models/z_image/padding.hpp']
        self.assertTrue(sources)
        for path in sources:
            source = path.read_text()
            for local_path in ['/Users/', '/home/', '../references/', '../notes/']:
                self.assertNotIn(local_path, source, str(path.relative_to(ROOT)))

    def test_acceleration_examples_are_portable_and_keep_lora_at_runtime(self):
        names = ['qwen21-base-512.json', 'qwen21-base-1024.json',
                 'z-image-turbo-1024.json',
                 'qwen21-viggle-runtime-lora-512.json',
                 'qwen21-viggle-runtime-lora-hybrid-512.json',
                 'z-image-gguf-base-512.json', 'z-image-runtime-lora-512.json']
        def check_paths(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('path', 'output', 'ane_manifest') and isinstance(item, str):
                        self.assertFalse(Path(item).is_absolute(), item)
                        self.assertNotIn('..', Path(item).parts, item)
                    check_paths(item)
            elif isinstance(value, list):
                for item in value:
                    check_paths(item)
        for name in names:
            with self.subTest(name=name):
                data = json.loads((ROOT/'examples/requests'/name).read_text())
                check_paths(data)
                self.assertNotIn('ane_manifest', data, 'supply local artifacts at invocation')
                if data.get('loras'):
                    self.assertEqual(data.get('lora_strategy'), 'inference_time')
                    self.assertNotEqual(data.get('hybrid_mlp_mode'), 'lora_merged')

    def test_acceleration_artifacts_stay_local(self):
        if not (ROOT/'.git').exists():
            self.skipTest('artifact tracking check requires a Git checkout')
        tracked = subprocess.check_output(
            ['git', 'ls-files', '--', 'results/z-image-*', 'results/qwen21/'],
            cwd=ROOT, text=True)
        self.assertEqual(tracked, '', 'keep portable examples and docs, not raw acceleration runs')
        paths = ['results/z-image-future-request.json',
                 'results/z-image-future.png',
                 'results/z-image-future-compiled/model/identity.json',
                 'results/qwen21/future/manifest.json']
        ignored = subprocess.check_output(
            ['git', 'check-ignore', '--no-index', '--stdin'],
            input='\n'.join(paths) + '\n', cwd=ROOT, text=True)
        self.assertEqual(ignored.splitlines(), paths)

    def test_single_implementation(self):
        for name in ['src','Sources','engines','scripts','model-packs','device-profiles','pyproject.toml','Package.swift','requirements-flux2.txt']:
            self.assertFalse((ROOT/name).exists(),name)
    def test_shipping_boundary(self):
        build=(ROOT/'tools/native/build.sh').read_text()
        self.assertNotIn('experimental/',build)
        self.assertIn('SOURCES=(',build)
        self.assertNotIn('find native',build)
        self.assertIn('h3_ane_disabled',build)
        h3_objc_line=next(
            line for line in build.splitlines()
            if line.startswith('for src in h3_metal '))
        for private_source in ['h3_ane_bridge','h3_ane_mlp','h3_ane_linear']:
            self.assertNotIn(private_source,h3_objc_line,
                             f'{private_source} must not be linked into shipping')
        disabled=(ROOT/'native/models/h3_runtime/h3_ane_disabled.c').read_text()
        self.assertIn('private ANE support is excluded',disabled)
        self.assertNotIn('_ANEInMemoryModel',disabled)
        self.assertFalse((ROOT/'native/models/h3_runtime/h3_ane_bridge.m').exists())
        self.assertFalse((ROOT/'native/models/h3_runtime/h3_ane_bridge.h').exists())
        # v2 permits ONE dynamically loaded private implementation, omitted
        # from the default public/distributable build. It must not leak into
        # shared runtime/model code; legacy H3 private bindings stay excluded.
        private_loader=ROOT/'native/backends/private/ane_program.mm'
        public_sources=build.split('SOURCES=(',1)[1].split('\n)',1)[0]
        self.assertNotIn('backends/private/',public_sources)
        self.assertIn('PRIVATE_ANE="${TURBOCIDER_ENABLE_PRIVATE_ANE:-0}"',build)
        private_gate=build.split('if [[ "$PRIVATE_ANE" == "1" ]]; then\n SOURCES+=(',1)[1].split('\nfi',1)[0]
        self.assertIn('native/backends/private/ane_program.mm',private_gate)
        loader=private_loader.read_text()
        self.assertIn('dlopen(',loader)
        self.assertIn('NSClassFromString',loader)
        for private_class in ['_ANEInMemoryModel','_ANEClient','_ANERequest','_ANEIOSurfaceObject','_ANESharedEvents']:
            for p in (ROOT/'native').rglob('*'):
                if p.suffix in ['.c','.cpp','.m','.mm','.h','.hpp']:
                    if p==private_loader and private_class!='_ANEInMemoryModel':
                        continue
                    self.assertNotIn(private_class,p.read_text(),str(p))
        release=(ROOT/'tools/native/check_release_binary.py').read_text()
        self.assertIn('TURBOCIDER_ENABLE_PRIVATE_ANE',release)
        self.assertIn("'strings'",release)
        self.assertIn("'otool'",release)
        for p in (ROOT/'native').rglob('*'):
            if p.suffix in ['.cpp','.mm','.hpp','.h']:
                self.assertNotIn('vendor/',p.read_text(),str(p))
    def test_product_entries(self):
        for name in ['apps/macos/App.swift','apps/cli/main.mm','services/turbociderd/service.mm','bindings/c/include/turbocider/turbocider.h','bindings/swift/TurboCiderNative.swift','profiles/apple-m4-pro-48gb.example.json','profiles/apple-m4-max-64gb.example.json']:
            self.assertTrue((ROOT/name).is_file(),name)
    def test_package_contains_video_runtime_resources(self):
        build=(ROOT/'tools/native/build.sh').read_text()
        self.assertIn('TURBOCIDER_DEPLOYMENT_TARGET',build)
        self.assertNotIn('mmacosx-version-min=15.0',build)
        self.assertNotIn('apple-macosx15.0',build)
        package=(ROOT/'tools/native/package.sh').read_text()
        for name in ['h3_shaders.metal','ltx_shaders.metal',
                     'h3-quantize-stream-cache',
                     'ltx-video-finalizer','ltx-video-vae-decode']:
            self.assertIn(name,package)
        self.assertNotIn('fastmetal_worker.py',package)
        self.assertIn('MLX_LICENSE_PATH',package)
        self.assertNotIn('mlx-0.32.0.dist-info',package)
        self.assertIn('DIST="${TURBOCIDER_PACKAGE_OUTPUT_DIR:-$ROOT/dist}"',package)
        self.assertIn('APP="$DIST/TurboCider.app"',package)
        self.assertIn('rm -rf "$APP" "$DIST/cli"',package)
        self.assertIn('<<PLIST',package)
        session=(ROOT/'native/platform/apple/ltx_session.mm').read_text()
        self.assertIn('dladdr((void*)&ltx_runtime_resource',session)
        self.assertIn('TURBOCIDER_RESOURCE_DIR',session)
if __name__=='__main__':unittest.main(verbosity=2)
