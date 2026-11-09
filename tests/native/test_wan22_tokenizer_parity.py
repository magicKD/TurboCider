"""UMT5 tokenizer versus the exact SentencePiece bytes in the Comfy checkpoint."""
import json
import os
from pathlib import Path
import random
import subprocess
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_UNIGRAM_PROBE", "/nonexistent")).resolve()
ROOT = Path(os.environ.get("TURBOCIDER_UMT5_TOKENIZER_ROOT", "/nonexistent")).resolve()
CHECKPOINT = Path(os.environ.get("TURBOCIDER_UMT5_CHECKPOINT", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file() and (ROOT / "tokenizer.json").is_file() and CHECKPOINT.is_file(),
                     "native tokenizer, UMT5 JSON and text checkpoint required")
class Wan22TokenizerParityTests(unittest.TestCase):
    def test_plain_prompts_against_embedded_sentencepiece(self):
        import sentencepiece
        from safetensors import safe_open

        with safe_open(str(CHECKPOINT), framework="pt", device="cpu") as checkpoint:
            proto = checkpoint.get_tensor("spiece_model").numpy().tobytes()
        oracle = sentencepiece.SentencePieceProcessor(model_proto=proto)
        prompts = ["A red fox walks slowly through the snow.", "一只狐狸在雪地中缓缓前行。",
                   "café and a kitten", "A  B", " A ", "  A", " ", "a\tb\nc",
                   "hello world", "🙂 in a garden", "<unk>", "</s>", "<extra_id_0>"]
        rng = random.Random(22)
        pieces = ["a", "fox", " ", "  ", "\t", "\n", "é", "e\u0301", "你", "，", "🙂", "<unk>"]
        prompts.extend("".join(rng.choice(pieces) for _ in range(rng.randrange(1, 12))) for _ in range(128))
        records = [{"prompt": prompt, "limit": 512} for prompt in prompts]
        result = subprocess.run([str(PROBE), str(ROOT)], input=json.dumps(records, ensure_ascii=False),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = json.loads(result.stdout)
        for prompt, actual in zip(prompts, outputs):
            with self.subTest(prompt=prompt):
                expected = oracle.encode(prompt, add_eos=True)
                self.assertEqual(actual["valid"], len(expected))
                self.assertEqual(actual["ids"], expected + [0] * (512 - len(expected)))

    def test_extended_prompt_preserves_all_tokens_and_rejects_overflow(self):
        import sentencepiece
        from safetensors import safe_open

        with safe_open(str(CHECKPOINT), framework="pt", device="cpu") as checkpoint:
            proto = checkpoint.get_tensor("spiece_model").numpy().tobytes()
        oracle = sentencepiece.SentencePieceProcessor(model_proto=proto)
        prompt = "A woman moves her sword slowly. " * 90
        expected = oracle.encode(prompt, add_eos=True)
        self.assertGreater(len(expected), 512)
        self.assertLessEqual(len(expected), 1024)
        result = subprocess.run([str(PROBE), str(ROOT)],
                                input=json.dumps([{"prompt": prompt, "extended": True}]),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)[0]
        self.assertEqual(actual["valid"], len(expected))
        self.assertEqual(actual["ids"], expected)
        overflow = "fox " * 2000
        self.assertGreater(len(oracle.encode(overflow, add_eos=True)), 1024)
        result = subprocess.run([str(PROBE), str(ROOT)],
                                input=json.dumps([{"prompt": overflow, "extended": True}]),
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceeds 1023", result.stderr)


if __name__ == "__main__":
    unittest.main()
