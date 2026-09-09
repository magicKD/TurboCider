"""Compare the native UMT5 Unigram tokenizer against tokenizer.json on CPU."""
import argparse
import json
from pathlib import Path
import random
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    a = p.parse_args()
    from tokenizers import Tokenizer
    reference = Tokenizer.from_file(str(a.tokenizer / 'tokenizer.json'))
    prompts = ['', ' ', '  hello  world ', 'A red fox walks across the snow.',
               '一只红狐狸走过雪地。', '日本語と한국어のテスト', 'مرحبا بالعالم',
               'é e\u0301 café', 'a\tb\nc\rd\u00a0e', '🫠🫥🤖', '\x00test',
               '<pad>', 'hello</s>world', '<extra_id_0>snow<extra_id_299>',
               'a▁b', '▁▁', '<unk><unk>', 'hello ' * 600,
               'x\ufeffy', 'a\x00b', '\\"quoted\\"', 'x\ufeff\ufeffy']
    rng = random.Random(42)
    pieces = ['a', 'hello', '世界', 'café', ' ', '  ', '\t', '\n', '🫠', '🫥',
              '<pad>', '</s>', '<extra_id_1>', '▁', '\u00a0', 'e\u0301']
    prompts.extend(''.join(rng.choices(pieces, k=rng.randint(1, 20))) for _ in range(200))
    records = [{'prompt': prompt, 'limit': limit} for prompt in prompts for limit in [1, 16, 512]]
    run = subprocess.run([str(a.probe.resolve()), str(a.tokenizer.resolve())],
                         input=json.dumps(records), text=True, capture_output=True)
    if run.returncode:
        raise RuntimeError(f'native tokenizer failed: {run.stderr}')
    results = json.loads(run.stdout)
    assert len(results) == len(records)
    failures = []
    for record, result in zip(records, results):
        reference.enable_truncation(max_length=record['limit'])
        reference.enable_padding(length=record['limit'], pad_id=0, pad_token='<pad>')
        encoded = reference.encode(record['prompt'])
        expected = {'ids': encoded.ids, 'valid': sum(encoded.attention_mask)}
        if result != expected:
            failures.append({'input': record, 'expected': expected, 'native': result})
    print(json.dumps({'cases': len(records), 'failed': len(failures), 'examples': failures[:5]}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
