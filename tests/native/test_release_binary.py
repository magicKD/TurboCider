import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))
import check_release_binary as release


class ReleaseBinaryTests(unittest.TestCase):
    def test_flags_and_exported_symbols_must_both_be_release(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            library = Path(folder) / 'lib.dylib'
            for hooks, counters in (('0', '0'), ('1', '0'), ('0', '1'), ('1', '1')):
                manifest.write_text(json.dumps({'inputs': {'policy': {'test_hooks': hooks, 'audit_counters': counters}}}))
                with patch.object(release.subprocess, 'check_output', return_value='0000 T _tc_engine_create\n'):
                    if hooks == counters == '0': release.check(library, manifest)
                    else:
                        with self.assertRaisesRegex(ValueError, 'disabled'): release.check(library, manifest)
            manifest.write_text(json.dumps({'inputs': {'policy': {'test_hooks': '0', 'audit_counters': '0'}}}))
            for symbol in ('_tc_engine_test_set_streaming_catalog_json', '_tc_streaming_audit_reset'):
                with patch.object(release.subprocess, 'check_output', return_value='0000 T ' + symbol + '\n'):
                    with self.assertRaisesRegex(ValueError, 'exposes instrumentation'):
                        release.check(library, manifest)

    def test_packaging_guard_precedes_existing_app_removal(self):
        source = (ROOT / 'tools/native/package.sh').read_text()
        self.assertLess(source.index('check_release_binary.py'), source.index('rm -rf "$APP"'))

    def test_stable_packaging_rejects_private_flags_actual_class_bytes_and_links(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            library = Path(folder) / 'lib.dylib'
            def policy(flags):
                manifest.write_text(json.dumps({'inputs': {'policy': {
                    'test_hooks': '0', 'audit_counters': '0', 'common_flags': flags}}}))
            for flags in (['-DTURBOCIDER_ENABLE_PRIVATE_ANE=1'], ['-DTURBOCIDER_ENABLE_PRIVATE_ANE'], None, [1]):
                policy(flags)
                with patch.object(release.subprocess, 'check_output') as inspect:
                    with self.assertRaises(ValueError): release.check(library, manifest)
                    inspect.assert_not_called()
            policy(['-DTURBOCIDER_ENABLE_PRIVATE_ANE=0'])
            def output(command, **kwargs):
                return {'nm': '0000 T _tc_engine_create\n', 'strings': '', 'otool': 'CoreML.framework\n'}[command[0]]
            with patch.object(release.subprocess, 'check_output', side_effect=output):
                release.check(library, manifest)
            for name in ('_ANEClient', '_ANERequest', '_ANEIOSurfaceObject', '_ANESharedEvents', '_ANEInMemoryModel'):
                with patch.object(release.subprocess, 'check_output', side_effect=lambda command, **kw:
                                  name if command[0] == 'strings' else output(command)):
                    with self.assertRaisesRegex(ValueError, 'private ANE class'): release.check(library, manifest)
            with patch.object(release.subprocess, 'check_output', side_effect=lambda command, **kw:
                              '/System/Library/PrivateFrameworks/AppleNeuralEngine.framework' if command[0] == 'otool' else output(command)):
                with self.assertRaisesRegex(ValueError, 'private framework'): release.check(library, manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
