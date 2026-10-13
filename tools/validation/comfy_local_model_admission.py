#!/usr/bin/env python3
"""Read-only stock loader admission for the original local MLX saved4B pack."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

from flux9_bf16_framework_screen import write_json
from mflux_original_layout import read_header
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def inspect(comfy, model):
    sys.path.insert(0, str(comfy))
    folders = importlib.import_module("folder_paths")
    transformer = model / "transformer"
    folders.add_model_folder_path("diffusion_models", str(transformer))
    choices = folders.get_filename_list("diffusion_models")
    index_path = transformer / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    names = sorted(set(index["weight_map"].values()))
    headers = {}
    for name in names:
        header, _ = read_header(transformer / name)
        for k, v in header.items():
            if k != "__metadata__":
                if k in headers:
                    raise ValueError("duplicate saved source parameter")
                headers[k] = v
    if set(headers) != set(index["weight_map"]):
        raise ValueError("complete original saved pack descriptors required")
    counts = {}
    for v in headers.values():
        counts[v["dtype"]] = counts.get(v["dtype"], 0) + 1
    return dict(schema="tc-comfy-local-format-admission-v1", date="2026-10-12", model="flux2-klein-4b",
        comfy_commit=subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip(),
        source="models/FLUX.2-klein-4B-mflux-Q4", source_index="model.safetensors.index.json",
        index_sha256=sha256_file(index_path), required_shards=names, required_parameters=len(headers), dtype_counts=counts,
        saved_metadata=index.get("metadata"), stock_unet_choices=choices, index_admitted=index_path.name in choices,
        compatible_complete_generation_verified=False, latency_comparison_valid=False,
        scope="metadata-only, no model execution/native retest or conversion; stock UNETLoader does not admit whole saved index; selectable single shards are not the complete387-parameter Q4 model; not a claim that ComfyUI lacks FLUX4B or that every custom plugin/alternative loader is unsupported",
        stock_source_sha256={str(p.relative_to(comfy)): sha256_file(p) for p in (comfy / "folder_paths.py", comfy / "nodes.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comfy", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh metadata evidence required")
    result = inspect(args.comfy.resolve(strict=True), args.model.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(index_admitted=result["index_admitted"], required_shards=result["required_shards"], parameters=result["required_parameters"])))


if __name__ == "__main__":
    main()
