"""T5 tokenizer parity with Hugging Face; no model weights or inference."""
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_UNIGRAM_PROBE", "/nonexistent")).resolve()
TOKENIZER_ROOT = Path(os.environ.get("TURBOCIDER_T5_TOKENIZER_ROOT", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file() and (TOKENIZER_ROOT / "tokenizer.json").is_file(),
                     "native tokenizer probe and T5 tokenizer fixture required")
class TokenizerParityTests(unittest.TestCase):
    def test_t5_matches_huggingface(self):
        from tokenizers import Tokenizer
        oracle = Tokenizer.from_file(str(TOKENIZER_ROOT / "tokenizer.json"))
        prompts = ["", " ", "A lighthouse stands on dark rocks as waves roll gently.",
                   " A  B  ", "hello  world", "a\tb\nc", "café", "cafe\u0301",
                   "ＡＢＣ  １２", "你好，世界", "🙂foo", "a<extra_id_0>b",
                   "<extra_id_0>abc", "<extra_id_0> abc", "abc</s>",
                   "a <extra_id_0>b", "  <extra_id_0>abc", "½ café", "hi▁world",
                   "A family 👨‍👩‍👧‍👦 walks through a garden.", "word " * 150]
        records = [{"prompt": prompt, "limit": 512} for prompt in prompts]
        run = subprocess.run([str(PROBE), str(TOKENIZER_ROOT), "--t5"],
                             input=json.dumps(records, ensure_ascii=False), capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        outputs = json.loads(run.stdout)
        self.assertEqual(len(outputs), len(prompts))
        for prompt, actual in zip(prompts, outputs):
            with self.subTest(prompt=prompt):
                expected = oracle.encode(prompt).ids
                self.assertEqual(actual["valid"], len(expected))
                self.assertEqual(actual["ids"], expected + [0] * max(0, 128 - len(expected)))

    def test_t5_rejects_prompt_over_limit(self):
        run = subprocess.run([str(PROBE), str(TOKENIZER_ROOT), "--t5"],
                             input=json.dumps([{"prompt": "word " * 600, "limit": 512}]),
                             capture_output=True, text=True, timeout=30)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("token limit", run.stderr)

    def test_public_google_tokenizer_uses_same_comfy_contract(self):
        from tokenizers import Tokenizer
        source = Path(os.environ.get("TURBOCIDER_T5_GOOGLE_TOKENIZER", "/nonexistent"))
        if not source.is_file():
            self.skipTest("public Google T5 tokenizer fixture required")
        oracle = Tokenizer.from_file(str(TOKENIZER_ROOT / "tokenizer.json"))
        prompts = [" A  B  ", "cafe\u0301", "ＡＢＣ  １２", "<extra_id_0>abc", "你好，世界"]
        with tempfile.TemporaryDirectory(prefix="tc-google-t5-") as temporary:
            root = Path(temporary)
            (root / "tokenizer.json").write_bytes(source.read_bytes())
            run = subprocess.run([str(PROBE), str(root), "--t5"],
                                 input=json.dumps([{"prompt": prompt, "limit": 512} for prompt in prompts], ensure_ascii=False),
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            for prompt, actual in zip(prompts, json.loads(run.stdout)):
                expected = oracle.encode(prompt).ids
                self.assertEqual(actual["ids"], expected + [0] * max(0, 128 - len(expected)))

    def test_mixed_unicode_and_special_token_corpus(self):
        from tokenizers import Tokenizer
        oracle = Tokenizer.from_file(str(TOKENIZER_ROOT / "tokenizer.json"))
        randomizer = random.Random(42)
        pieces = ["a", "words", " ", "  ", "\n", "\t", "café", "e\u0301", "Ａ",
                  "你", "好", "，", "🙂", "▁", "<extra_id_0>", "<extra_id_1>", "!</s>"]
        prompts = ["".join(randomizer.choice(pieces) for _ in range(randomizer.randrange(1, 18)))
                   for _ in range(256)]
        records = [{"prompt": prompt, "limit": 512} for prompt in prompts]
        run = subprocess.run([str(PROBE), str(TOKENIZER_ROOT), "--t5"],
                             input=json.dumps(records, ensure_ascii=False), capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        for prompt, actual in zip(prompts, json.loads(run.stdout)):
            with self.subTest(prompt=prompt):
                expected = oracle.encode(prompt).ids
                self.assertEqual(actual["valid"], len(expected))
                self.assertEqual(actual["ids"], expected + [0] * max(0, 128 - len(expected)))

    def test_existing_umt5_fixture_keeps_fixed_padding(self):
        tokenizer = {
            "model": {"type": "Unigram", "unk_id": 3, "byte_fallback": False,
                      "vocab": [["<pad>", 0], ["</s>", 0], ["<s>", 0], ["<unk>", 0],
                                ["▁", -1], ["a", -2], ["▁a", -.5]]},
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]},
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
            "post_processor": {"type": "TemplateProcessing", "single": [
                {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                "special_tokens": {"</s>": {"ids": [1]}}},
            "added_tokens": []}
        with tempfile.TemporaryDirectory(prefix="tc-umt5-tokenizer-") as temporary:
            root = Path(temporary)
            (root / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False))
            run = subprocess.run([str(PROBE), str(root)], input=json.dumps([{"prompt": "a", "limit": 8}]),
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), [{"ids": [6, 1, 0, 0, 0, 0, 0, 0], "valid": 2}])


if __name__ == "__main__":
    unittest.main()
