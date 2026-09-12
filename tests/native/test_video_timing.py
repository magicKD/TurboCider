"""Verify actual native-writer MP4 timing, not rounded nominal FPS."""
import json
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class VideoTimingTests(unittest.TestCase):
    def test_exact_integer_fps_and_duration(self):
        if not shutil.which('ffprobe') or not shutil.which('xcrun'):
            self.skipTest('ffprobe and Apple development tools required')
        library = ROOT / 'build/native'
        sdk = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'],
                                      text=True).strip()
        # Match the deployment target required by the native library (MLX can
        # be built for a newer macOS than clang's default test target).
        load_commands = subprocess.check_output(
            ['otool', '-l', str(library / 'libturbocider.dylib')], text=True
        )
        deployment_target = None
        build_version = False
        for line in load_commands.splitlines():
            if line.strip() == 'cmd LC_BUILD_VERSION':
                build_version = True
            elif build_version and line.strip().startswith('minos '):
                deployment_target = line.strip().split()[1]
                break
        if deployment_target is None:
            self.fail('could not determine native library deployment target')
        with tempfile.TemporaryDirectory(prefix='tc-video-timing-') as directory:
            probe = Path(directory) / 'probe'
            subprocess.run(['xcrun', 'clang++', '-std=c++20', '-isysroot', sdk,
                            '-mmacosx-version-min=' + deployment_target,
                            str(ROOT / 'tests/native/video_timing_probe.cpp'),
                            '-L' + str(library), '-lturbocider',
                            '-Wl,-rpath,' + str(library), '-o', str(probe)], check=True)
            subprocess.run([str(probe), directory], check=True, timeout=120)
            for fps in [16, 24, 30]:
                for frames in [5, 81]:
                    with self.subTest(fps=fps, frames=frames):
                        result = subprocess.check_output([
                            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=avg_frame_rate,duration_ts,time_base,nb_frames',
                            '-of', 'json', str(Path(directory) / f'{fps}-{frames}.mp4')], text=True)
                        stream = json.loads(result)['streams'][0]
                        self.assertEqual(Fraction(stream['avg_frame_rate']), fps)
                        self.assertEqual(int(stream['nb_frames']), frames)
                        self.assertEqual(int(stream['duration_ts']) * Fraction(stream['time_base']),
                                         Fraction(frames, fps))


if __name__ == '__main__':
    unittest.main()
