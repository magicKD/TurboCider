"""Prevent platform/tensor dependencies from leaking back into generic runtime."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2]
class CppBoundaries(unittest.TestCase):
    def test_core_and_models_have_no_objc(self):
        for folder in ['native/core','native/runtime','native/models']:
            for p in (ROOT/folder).rglob('*'):
                if not p.is_file():continue
                self.assertNotEqual(p.suffix,'.mm',str(p))
                if p.suffix in ['.hpp','.cpp']:
                    for token in ['Foundation/','CoreML/CoreML','NSDictionary','NSProcessInfo','@"']:
                        self.assertNotIn(token,p.read_text(),str(p))
        for folder in ['native/core','native/runtime']:
            for p in (ROOT/folder).glob('*.hpp'):
                self.assertNotIn('mlx/',p.read_text(),str(p))
    def test_cpu_only_runtime(self):
        with tempfile.TemporaryDirectory(prefix='tc-cpp-policy-') as directory:
            binary=str(Path(directory)/'test')
            subprocess.run(['clang++','-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT/'native'),'-I',str(ROOT/'native/core'),str(ROOT/'tests/native/runtime_policy_test.cpp'),str(ROOT/'native/core/common.cpp'),str(ROOT/'native/runtime/residency.cpp'),'-o',binary],check=True)
            subprocess.run([binary],check=True)
if __name__=='__main__':unittest.main(verbosity=2)
