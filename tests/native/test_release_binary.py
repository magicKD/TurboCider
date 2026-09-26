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


if __name__ == '__main__':
    unittest.main(verbosity=2)
