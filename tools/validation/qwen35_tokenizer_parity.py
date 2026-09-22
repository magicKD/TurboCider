"""Generate official HF tokenizer cases for the native BPE decode/encode probe."""
import argparse
import json
from pathlib import Path
import subprocess
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    texts = ["Hello world!", "你好，透明茶壶 🫖🌈", "cafe\u0301 résumé", "\t line one\r\nline two  ",
             "1234567890 1.25 -2e10", "\x00 control \x01 bytes", "العربية हिन्दी 日本語 한국어",
             '<think>\nplan\n</think>\n{"rewritten_prompt":"你好","wh_ratio":"1:1"}',
             "<|im_start|>assistant\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>",
             "a red teapot " * 1024]
    cases = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append(dict(text=text, ids=ids, decoded=tokenizer.decode(ids, skip_special_tokens=False,
                                                                     clean_up_tokenization_spaces=False)))
    if (args.tokenizer / "chat_template.jinja").is_file():
        for references in (0, 1, 10):
            system, prompt = " \u2003System 指令\n", " \u2003红色茶壶  \n"
            messages = [{"role":"system", "content": [{"type":"text", "text":system}]},
                        {"role":"user", "content": [{"type":"image"} for _ in range(references)] +
                            [{"type":"text", "text":prompt}]}]
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            ids = tokenizer.encode(chat, add_special_tokens=False)
            cases.append(dict(text=chat, ids=ids, decoded=tokenizer.decode(ids, skip_special_tokens=False),
                              system=system, prompt=prompt, references=references))
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(cases, ensure_ascii=False))
    subprocess.run([str(args.probe.resolve()), str(args.tokenizer.resolve()), str(args.fixture.resolve())], check=True)


if __name__ == "__main__":
    main()
