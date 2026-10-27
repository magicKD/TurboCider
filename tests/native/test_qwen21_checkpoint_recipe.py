"""CPU-only source selection/Turbo recipe and pinned ModelScope controls."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class CheckpointRecipeTests(unittest.TestCase):
    def test_native_source_and_turbo_route_controls(self):
        with tempfile.TemporaryDirectory(prefix='tc-qwen21-recipe-') as d:
            binary=Path(d)/'probe'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'native/core'),
                str(ROOT/'tests/native/qwen21_checkpoint_recipe_test.cpp'),'-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary),d],check=True,cwd=ROOT)

    def test_modelscope_pins_distinguish_turbo_and_convrot(self):
        spec=importlib.util.spec_from_file_location('qwen21_ms',ROOT/'tools/models/download_qwen21_modelscope.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.REPOSITORY,'Comfy-Org/Qwen-Image-2.1');self.assertEqual(len(module.PINS),4)
        revision,path,size,digest=module.PINS['turbo-bf16'];self.assertEqual(size,14230280616)
        self.assertEqual(digest,'8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a')
        self.assertTrue(path.endswith('qwen_image_2.1_turbo_bf16.safetensors'));self.assertEqual(len(revision),40)
        self.assertEqual(len({p[3] for p in module.PINS.values()}),4)

    def test_consumed_source_not_hardcoded_base_in_identity_or_calibration(self):
        source=(ROOT/'native/models/qwen21/pipeline.cpp').read_text()
        self.assertNotIn('root_ / "diffusion_models/qwen_image_2.1_bf16.safetensors"',source)
        self.assertNotIn('root_/"diffusion_models/qwen_image_2.1_bf16.safetensors"',source)
        self.assertIn('turbo ? turbo_sigmas',source);self.assertIn('transformer_checkpoint_source(root)',source)
        self.assertLess(source.index('validate_checkpoint_recipe(requested'),source.index('if(bf16_streaming_enabled())',source.index('RunResult Session::run(')))


if __name__=='__main__':unittest.main()
