#!/usr/bin/env python3
"""Export fixed-shape FastMetal FFN Core ML artifacts for TurboCider."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SCHEMA = "turbocider-fastmetal-ane-mlp-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--ane-width", type=int, required=True)
    parser.add_argument("--variant", choices=("fp16", "int8_pc"), default="int8_pc")
    parser.add_argument("--blocks", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--keep-packages", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("rows must be positive")
    if not 0 < args.ane_width < 8960 or args.ane_width % 64:
        parser.error("ane-width must be a 64-aligned proper FFN shard")
    if sorted(set(args.blocks)) != list(range(30)):
        parser.error("production FastMetal ANE export currently requires blocks 0..29")
    return args


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from benchmark_fastmetal_ane_mlp import (
        HIDDEN,
        INTERMEDIATE,
        compiled_artifact,
        dequantized_prefix,
        load_packed_mlp,
        sha256_file,
    )
    import mlx.core as mx

    model_root = args.model_root.expanduser().resolve()
    checkpoint = model_root / "mlx_dit.safetensors"
    config_path = model_root / "mlx_dit.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise SystemExit(
            f"{model_root} must contain mlx_dit.safetensors and mlx_dit.json"
        )
    packed_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = packed_config.get("config", packed_config)
    expected = {
        "num_layers": 30,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "ffn_dim": INTERMEDIATE,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if int(config.get(key, -1)) != value
    }
    if mismatches:
        raise SystemExit(f"unsupported FastMetal architecture: {mismatches}")
    if int(config["num_attention_heads"]) * int(config["attention_head_dim"]) != HIDDEN:
        raise SystemExit("unsupported FastMetal hidden size")

    checkpoint_digest = sha256_file(checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for block in args.blocks:
        print(f"[{block + 1}/30] exporting FastMetal block {block}", flush=True)
        packed = load_packed_mlp(checkpoint, block, mx)
        dense_prefix = dequantized_prefix(packed, args.ane_width, mx)
        artifact = compiled_artifact(
            artifact_dir=output_dir,
            block=block,
            checkpoint_digest=checkpoint_digest,
            rows=args.rows,
            ane_width=args.ane_width,
            variant=args.variant,
            weights=dense_prefix,
            force=args.force,
        )
        artifacts[str(block)] = artifact.name
        if not args.keep_packages:
            shutil.rmtree(artifact.with_suffix(".mlpackage"), ignore_errors=True)
        del packed, dense_prefix
        mx.clear_cache()

    manifest = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "mlx_dit_json_sha256": sha256_file(config_path),
        "shape": {
            "rows": args.rows,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "ane_intermediate": args.ane_width,
            "gpu_intermediate": INTERMEDIATE - args.ane_width,
        },
        "variant": args.variant,
        "blocks": args.blocks,
        "artifacts": artifacts,
        "tensor_abi": {
            "logical_input": [1, args.rows, HIDDEN],
            "coreml_input": [1, HIDDEN, 1, args.rows],
            "logical_output": [1, args.rows, HIDDEN],
            "dtype": "float16",
            "output_backing": "caller-owned-mlx-buffer",
        },
        "device_policy": {
            "validated_profiles": ["apple-m4-max-40gpu-64gb"],
            "explicit_force_allowed": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
