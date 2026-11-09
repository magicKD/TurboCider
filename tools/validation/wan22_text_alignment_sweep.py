#!/usr/bin/env python3
"""Compare original and aligned Wan2.2 text encoding using real weights."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alignment", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 10, 32, 53, 64, 65, 127, 128, 129, 191, 255, 256, 257, 511, 512, 526])
    parser.add_argument("--random-token-seed", type=int, help="test random valid vocabulary IDs rather than captured content")
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir() or any(n < 1 or n > 1024 for n in args.lengths):
        parser.error("output must be new and lengths must be 1...1024")
    probe, checkpoint, tokenizer, capture = (p.resolve(strict=True) for p in
                                               (args.probe, args.checkpoint, args.tokenizer, args.capture))
    source = load_file(str(capture))
    content = source["ids"][:int(source["valid"].item()) - 1]
    if content.numel() == 0:
        raise ValueError("capture has no content tokens")
    report = {"schema": "wan22-text-alignment-sweep-v1", "alignment": args.alignment,
              "scope": "real UMT5 weights; captured or random vocabulary IDs; separate fresh GPU processes",
              "random_token_seed": args.random_token_seed, "cases": []}
    generator = torch.Generator(device="cpu").manual_seed(args.random_token_seed) if args.random_token_seed is not None else None
    with tempfile.TemporaryDirectory(prefix="tc-wan22-text-alignment-") as temporary:
        root = Path(temporary)
        for valid in args.lengths:
            token_ids = (torch.randint(4, 256000, (valid - 1,), generator=generator, dtype=torch.int32) if generator is not None
                         else content.repeat((valid + content.numel() - 1) // content.numel())[:valid - 1])
            ids = torch.cat((token_ids, torch.tensor([1], dtype=torch.int32)))
            if ids.numel() < 512:
                ids = torch.cat((ids, torch.zeros(512 - ids.numel(), dtype=torch.int32)))
            fixture = root / f"input-{valid}.safetensors"
            save_file({"ids": ids.contiguous(), "valid": torch.tensor(valid, dtype=torch.int32)}, str(fixture))
            outputs, times = [], []
            for label, flag in (("baseline", []), ("aligned", [f"--align{args.alignment}"])):
                destination = root / f"{label}-{valid}.safetensors"
                run = subprocess.run([str(probe), str(checkpoint), str(tokenizer), str(fixture), str(destination), *flag],
                                     capture_output=True, text=True, timeout=180, check=True)
                match = re.search(r"text_encode_seconds=([0-9.eE+-]+)", run.stderr)
                if not match:
                    raise ValueError("probe did not report encoding time")
                times.append(float(match.group(1)))
                outputs.append(load_file(str(destination))["output"].float())
            difference = outputs[0] - outputs[1]
            norm = outputs[0].norm().clamp_min(1e-30)
            case = {"valid_tokens": valid, "sequence_tokens": int(ids.numel()),
                    "baseline_seconds": times[0], "aligned_seconds": times[1],
                    "equal": torch.equal(outputs[0], outputs[1]), "changed_values": int(torch.count_nonzero(difference)),
                    "max_abs": difference.abs().max().item(), "relative_l2": (difference.norm() / norm).item()}
            report["cases"].append(case)
            print(json.dumps(case, sort_keys=True), flush=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
