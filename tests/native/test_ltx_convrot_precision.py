"""Exercise hardware selection and explicit comparison routes without a GPU."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LTXPrecisionTests(unittest.TestCase):
    def test_hardware_and_overrides(self):
        source = r'''
#include "native/models/ltx_mlx/precision.hpp"
#include <cassert>
using tc::ltx_mlx::configured_convrot_w8a8;
int main() {
    unsetenv("TURBOCIDER_LTX_CONVROT_W8A8");
    unsetenv("TURBOCIDER_LTX_RAW_CONVROT");
    assert(!configured_convrot_w8a8(false));
    assert(configured_convrot_w8a8(true));
    setenv("TURBOCIDER_LTX_CONVROT_W8A8", "0", 1);
    assert(!configured_convrot_w8a8(true));
    setenv("TURBOCIDER_LTX_CONVROT_W8A8", "1", 1);
    bool rejected = false;
    try { configured_convrot_w8a8(false); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    setenv("TURBOCIDER_LTX_RAW_CONVROT", "1", 1);
    rejected = false;
    try { configured_convrot_w8a8(true); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    unsetenv("TURBOCIDER_LTX_CONVROT_W8A8");
    assert(!configured_convrot_w8a8(true));
    setenv("TURBOCIDER_LTX_CONVROT_W8A8", "invalid", 1);
    rejected = false;
    try { configured_convrot_w8a8(true); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'policy.cpp').write_text(source)
            subprocess.run(['clang++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT), str(path / 'policy.cpp'), '-o', str(path / 'policy')], check=True)
            subprocess.run([str(path / 'policy')], check=True)


if __name__ == '__main__':
    unittest.main()
