"""Prepare a read-only-linked HF view and convert the identical Qwen checkpoint.

Reference source is never changed. Output GGUF is a benchmark artifact, not a
TurboCider-supported text component. Conversion uses upstream llama.cpp code.
"""
import argparse
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-source", type=Path, required=True)
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    hf = a.output / "hf"
    hf.mkdir()
    (hf / "config.json").symlink_to(a.config.resolve())
    (hf / "model.safetensors").symlink_to(a.components.resolve() / "split_files/text_encoders/qwen_3_4b.safetensors")
    for file in (a.components.resolve() / "tokenizer").iterdir():
        if file.is_file(): (hf / file.name).symlink_to(file)
    subprocess.run([sys.executable, str(a.llama_source / "convert_hf_to_gguf.py"), str(hf),
                    "--outtype", "bf16", "--outfile", str(a.output / "qwen3-bf16.gguf")], check=True)
