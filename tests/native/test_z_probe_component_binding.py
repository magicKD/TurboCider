"""Local component content/generation binding without a model runtime."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("z_probe_identity", ROOT/"tools/native/z_image_probe_identity.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ComponentBindingTests(unittest.TestCase):
    def test_native_comfy_precedence_and_content_generation(self):
        with tempfile.TemporaryDirectory(prefix="tc-z-binding-") as raw:
            root = Path(raw)
            for role in ("split_files/vae", "split_files/text_encoders", "tokenizer", "text_encoder"):
                (root/role).mkdir(parents=True)
            (root/"split_files/vae/ae.safetensors").write_bytes(b"vae fixture")
            text = root/"split_files/text_encoders/qwen_3_4b.safetensors"
            text.write_bytes(b"Comfy text fixture")
            shard = root/"text_encoder/model.safetensors"; shard.write_bytes(b"other text fixture")
            tokenizer = root/"tokenizer/tokenizer.json"; tokenizer.write_bytes(b"tokenizer fixture")
            gpu, states, _ = module.bind_components(root,True,{})
            quant, _, _ = module.bind_components(root,False,{})
            self.assertNotEqual(gpu["encoder"],quant["encoder"])
            self.assertEqual(gpu["tokenizer"],quant["tokenizer"])
            module.revalidate_components(states)
            shard.write_bytes(b"changed text fixture")
            with self.assertRaises(ValueError): module.revalidate_components(states)

    def test_symlink_retarget_is_not_same_generation(self):
        with tempfile.TemporaryDirectory(prefix="tc-z-binding-") as raw:
            root = Path(raw); a=root/"a"; b=root/"b"; link=root/"component"
            a.write_bytes(b"same content"); b.write_bytes(b"same content"); link.symlink_to(a)
            states={link:module.snapshot(link)}
            link.unlink(); link.symlink_to(b)
            with self.assertRaises(ValueError): module.revalidate_components(states)

    def test_multiple_convenience_links_are_ambiguous(self):
        with tempfile.TemporaryDirectory(prefix="tc-z-binding-") as raw:
            root = Path(raw); folder=root/"shards"; folder.mkdir(); file=root/"weight"; file.write_bytes(b"fixture")
            for name in ("a", "b"): (folder/(name+".safetensors")).symlink_to(file)
            with self.assertRaises(ValueError): module.safetensors(folder)


if __name__=="__main__": unittest.main()
