import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/validation"))
from seedvr2_framework_screen import validate_native_build_binding


class NativeBuildBinding(unittest.TestCase):
    def test_packaged_bytes_must_match_the_explicit_identity(self):
        storage=C.create_string_buffer(b"tc-runtime-build-v1-a")
        pointer=C.cast(storage,C.c_void_p).value
        loaded=SimpleNamespace(tc_runtime_build_identity=Mock(return_value=pointer),tc_string_free=Mock())
        with patch.object(C,"CDLL",return_value=loaded):
            self.assertEqual(validate_native_build_binding(Path("packaged.dylib"),dict(runtime_build_id="tc-runtime-build-v1-a")),"tc-runtime-build-v1-a")
            for identity in ({},dict(runtime_build_id="tc-runtime-build-v1-b"),dict(runtime_build_id=1)):
                with self.assertRaises(ValueError):validate_native_build_binding(Path("packaged.dylib"),identity)
        self.assertEqual(loaded.tc_string_free.call_count,4)
        loaded.tc_string_free.assert_called_with(pointer)

    def test_missing_runtime_identity_is_not_bypassed(self):
        loaded=SimpleNamespace(tc_runtime_build_identity=Mock(return_value=None),tc_string_free=Mock())
        with patch.object(C,"CDLL",return_value=loaded),self.assertRaises(ValueError):
            validate_native_build_binding(Path("packaged.dylib"),dict(runtime_build_id="x"))
        loaded.tc_string_free.assert_not_called()


if __name__=="__main__":unittest.main()
