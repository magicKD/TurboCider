"""Promotion must preserve the already-screened kernel and typed boundaries."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class SwigluNativeSourceTests(unittest.TestCase):
    def test_promoted_helper_is_the_same_screened_expression(self):
        lab=(ROOT/"tools/native/qwen21_lora_swiglu_prepare.hpp").read_text().strip()
        native=(ROOT/"native/models/qwen21/metal/lora_swiglu.hpp").read_text().strip()
        native=native.replace("// Default-off native-base", "// Research-only native-base")
        native=native.replace("namespace tc::qwen21::metal {","namespace tc::qwen21::swiglu_lab {")
        native=native.replace("inline mx::array prepare_lora_swiglu(","inline mx::array finish(")
        self.assertEqual(native,lab)


if __name__=="__main__":unittest.main()
