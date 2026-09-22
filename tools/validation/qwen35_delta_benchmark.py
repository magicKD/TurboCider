"""Synthetic native GPU delta benchmark. Development-only; not PE model speed."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cases = []
    with tempfile.TemporaryDirectory(prefix="tc-qwen35-delta-benchmark-") as directory:
        root = Path(directory)
        for length in (128, 512, 2450):
            data = {"q": torch.randn(1, length, 32, 128).bfloat16(),
                    "k": torch.randn(1, length, 32, 128).bfloat16(),
                    "v": torch.randn(1, length, 32, 128).bfloat16(),
                    "g": -torch.rand(1, length, 32),
                    "beta": torch.rand(1, length, 32).bfloat16()}
            save_file(data, str(root / "input.safetensors"),
                      {"normalize": "true", "chunk_size": "64", "benchmark": "true"})
            subprocess.run([str(args.probe.resolve()), str(root / "input.safetensors"),
                            str(root / "output.safetensors"), "--gpu"], check=True)
            result = load_file(str(root / "output.safetensors"))
            torch.testing.assert_close(result["chunk_output"], result["output"], atol=.004, rtol=.004)
            torch.testing.assert_close(result["chunk_state"], result["state"], atol=3e-5, rtol=3e-5)
            baseline, candidate = result["seconds"].tolist()
            record = dict(length=length, heads=32, key_dim=128, value_dim=128,
                          recurrent_seconds=baseline, chunked_seconds=candidate,
                          speedup=statistics.median(baseline)/statistics.median(candidate))
            cases.append(record)
            print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps(dict(
        scope="Synthetic GPU primitive, one unrecorded warmup then three runs; NOT whole PE speedup",
        cases=cases), indent=2) + "\n")


if __name__ == "__main__":
    main()
