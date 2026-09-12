#!/usr/bin/env python3
"""Compare TurboCider raw BPE IDs with Transformers add_special_tokens=False."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROMPTS = (
    "A red fox runs through fresh snow.",
    "一只橘猫跳上窗台，晨光照亮它的胡须。",
    "The singer says: \"Hello, world!\" <|im_end|>",
    "Café déjà vu — こんにちは 👋🏽",
    "line one\nline two\t12345",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    reference = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    for prompt in PROMPTS:
        expected = [int(value) for value in reference(prompt, add_special_tokens=False)["input_ids"]]
        actual = json.loads(subprocess.check_output(
            [str(args.probe), str(args.tokenizer), prompt], text=True
        ))
        if actual != expected:
            raise AssertionError(
                f"tokenizer mismatch for {prompt!r}\nexpected={expected}\nactual={actual}"
            )
    print(f"PASS: native H3 raw tokenizer parity ({len(PROMPTS)} prompts)")


if __name__ == "__main__":
    main()
