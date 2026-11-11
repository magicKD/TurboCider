"""A package refresh must not leave the desktop running deleted old binaries."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('check_package_in_use', ROOT / 'tools/native/check_package_in_use.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class PackageInUseTests(unittest.TestCase):
    def test_app_and_cli_workers_with_spaces_but_not_neighboring_packages(self):
        app = Path('/tmp/Turbo Cider/TurboCider.app')
        cli = app.parent / 'cli'
        processes = f'''  10 {app}/Contents/MacOS/TurboCiderNativeApp
 11 {app}/Contents/MacOS/ltx-video-finalizer
 12 {cli}/turbocider
 13 {app}.backup/Contents/MacOS/TurboCiderNativeApp
 14 {cli}-old/turbocider
 15 python3
'''
        self.assertEqual([pid for pid, _ in CHECK.package_processes(app, cli, processes)], [10, 11, 12])

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS executable identity uses libproc')
    def test_detects_real_process_paths(self):
        listing = CHECK.running_processes()
        own = next(line for line in listing.splitlines() if line.strip().split(None, 1)[0] == str(os.getpid()))
        executable = Path(own.strip().split(None, 1)[1])
        self.assertTrue(executable.is_absolute())
        self.assertEqual(len(CHECK.package_processes(executable.parent, Path('/unused-tc-cli'), own)), 1)

    def test_command_refuses_stale_process_path_even_if_executable_is_gone(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            app = root / 'TurboCider.app'
            app.mkdir()
            marker = app / 'keep'
            marker.write_text('existing package')
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            ps = bin_dir / 'ps'
            process = f'42 {app}/Contents/MacOS/TurboCiderNativeApp'
            ps.write_text(f'#!{sys.executable}\nprint({process!r})\n')
            ps.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / 'tools/native/check_package_in_use.py'),
                                     '--app', str(app), '--cli', str(root / 'cli')],
                                    env={**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.defpath},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('PID 42:', result.stderr)
            self.assertEqual(marker.read_text(), 'existing package')

    def test_symlinked_package_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            app = root / 'TurboCider.app'
            app.mkdir()
            alias = root / 'Alias.app'
            alias.symlink_to(app, target_is_directory=True)
            matches = CHECK.package_processes(alias, root / 'cli', f'20 {app}/Contents/MacOS/TurboCiderNativeApp')
            self.assertEqual([pid for pid, _ in matches], [20])

    def test_guard_precedes_package_deletion(self):
        source = (ROOT / 'tools/native/package.sh').read_text()
        self.assertLess(source.index('tools/native/check_package_in_use.py'), source.index('rm -rf "$APP" "$DIST/cli"'))


if __name__ == '__main__':
    unittest.main()
