"""Scratch allocation contract and route-layout comparison checks."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from benchmark_ltx_route_layout import equal_metrics


class RouteLayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("clang"), "clang needed for C allocation contract")
    def test_actual_c_scratch_size_helper(self):
        source = r'''
#include "ltx_gpu.h"
#include <assert.h>
int main(void) {
    assert(ltx_sparse_route_scratch_words(0, 32) == 0);
    assert(ltx_sparse_route_scratch_words(1, 0) == 0);
    assert(ltx_sparse_route_scratch_words(UINT32_MAX, 32) == 0);
    assert(ltx_sparse_route_scratch_words(16385, 32) == 0);
    assert(ltx_sparse_route_scratch_words(1, 2) == 512);
    assert(ltx_sparse_route_scratch_words(5376, 32) * 4 == 32768);
    assert(ltx_sparse_route_scratch_words(14080, 32) * 4 == 197120);
    assert(ltx_sparse_route_scratch_words(16384, 32) * 4 == 262144);
    for (unsigned rows = 1; rows <= 16384; ++rows) {
        unsigned blocks = (rows + 63) / 64;
        unsigned long long words = ltx_sparse_route_scratch_words(rows, 32);
        assert(words >= (unsigned long long)32 * blocks * ((blocks + 31) / 32));
        assert(words >= 32 * 128 * 2);
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "route-layout-test")
            subprocess.run([shutil.which("clang"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-x", "c", "-", "-I", str(ROOT / "native/models/ltx_runtime"),
                            "-o", binary], input=source, text=True, capture_output=True, check=True)
            subprocess.run([binary], capture_output=True, check=True)

    def test_comparator_checks_per_head_not_just_global(self):
        metrics = dict(relative_l2=.1, cosine=.99, max_abs=1., exact_block_fraction=.5,
                       per_head=[dict(head=0, squared_error=1.)])
        other = deepcopy(metrics)
        self.assertTrue(equal_metrics(metrics, other))
        other['per_head'][0]['squared_error'] = 2.
        self.assertFalse(equal_metrics(metrics, other))
