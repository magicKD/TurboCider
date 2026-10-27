"""Host-only capability oracle negative controls, no ANE/model work."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/profiling/qwen_base'))
spec = importlib.util.spec_from_file_location('multi_function_collect', ROOT / 'tools/profiling/qwen_base/multi_function_collect.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class MultiFunctionOracleTests(unittest.TestCase):
    def arrays(self):
        x = np.arange(128*64, dtype=np.float16).reshape(128,64)/np.float16(128)
        w = np.eye(128, dtype=np.float16)*np.float16(.5)
        y = np.full_like(x, 1234)
        y[:, :32] = x[:, :32]*np.float16(.5)
        return x,w,y

    def test_full_contraction_and_padding(self):
        self.assertEqual(screen.full_array_oracle(*self.arrays(),32),(0.,True))

    def test_nonfinite_source_or_output_never_passes(self):
        for target in range(3):
            for bad in (np.nan,np.inf,-np.inf):
                values = self.arrays()
                values[target][0,0] = bad
                with self.assertRaises(ValueError): screen.full_array_oracle(*values,32)

    def test_wrong_value_tail_and_shape_fail(self):
        for kind in ('value','tail','shape'):
            x,w,y = self.arrays()
            if kind == 'value': y[0,0] = 1
            if kind == 'tail': y[0,32] = 0
            if kind == 'shape': x = x[:64]
            with self.assertRaises(ValueError): screen.full_array_oracle(x,w,y,32)


if __name__ == '__main__':
    unittest.main()
