#!/usr/bin/env python3
"""Export identity-bound SeedVR2 video-FFN prefixes for native GPU+ANE.

The GPU owns the remaining SwiGLU intermediate channels. This branch contains
the gate/up prefix and matching down columns; it has no residual or gate from
the transformer block. The stable profile retains original FP16 coefficients;
BF16-rounded operands are an explicit legacy diagnostic option. Core ML uses
FP16 activation math, not weight quantization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from export_ltx_gemma_mlp import mapped_tensor, read_header, round_bf16, tensor_info

SCHEMA = "turbocider-seedvr2-ffn-prefix-v1"
CHECKPOINT_SHA = "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660"
HIDDEN, INTERMEDIATE, LAYERS = 2560, 6912, 32
OPERANDS = {"original-fp16": "original-fp16", "bf16-rounded": "original-fp16-rounded-to-bf16-then-fp16"}


def prepare_operand(value, recipe):
    import numpy as np
    if recipe not in OPERANDS:
        raise ValueError("unknown SeedVR2 operand recipe")
    if recipe == "bf16-rounded":
        value = round_bf16(value)
    return np.ascontiguousarray(value, dtype=np.float16)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(root):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path)))
    if not files:
        raise ValueError("empty compiled artifact")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return dict(files=files, sha256=hashlib.sha256(encoded).hexdigest())


def parse_blocks(value):
    result = set()
    for token in value.split(","):
        if "-" in token:
            first, last = map(int, token.split("-"))
            if not 0 <= first <= last < LAYERS:
                raise ValueError("block ranges must be ordered within 0...31")
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    if not result or min(result) < 0 or max(result) >= LAYERS:
        raise ValueError("blocks must be in 0...31")
    return sorted(result)


def build_model(rows, width, gate, up, down, operand_recipe):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, rows), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x):
        g = mb.conv(x=x, weight=gate[:, :, None, None], pad_type="valid", name="gate")
        u = mb.conv(x=x, weight=up[:, :, None, None], pad_type="valid", name="up")
        activated = mb.mul(x=g, y=mb.sigmoid(x=g), name="silu")
        product = mb.mul(x=activated, y=u, name="product")
        return mb.conv(x=product, weight=down[:, :, None, None], pad_type="valid", name="y")

    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    model.user_defined_metadata.update({"turbocider.schema": SCHEMA, "turbocider.activation": "silu-gated",
        "turbocider.source_operand_recipe": operand_recipe,
        "turbocider.prefix_channels": str(width), "turbocider.rows": str(rows)})
    return model


def export(args):
    import coremltools as ct
    import numpy as np
    checkpoint = args.checkpoint.resolve(strict=True)
    actual = sha256(checkpoint)
    if actual != CHECKPOINT_SHA:
        raise ValueError("original Comfy-Org SeedVR2 FP16 checkpoint SHA-256 required")
    offset, header = read_header(checkpoint)
    blocks = parse_blocks(args.blocks)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    configuration = dict(schema=SCHEMA, checkpoint_sha256=actual, checkpoint_bytes=checkpoint.stat().st_size,
        rows=args.rows, hidden=HIDDEN, intermediate=INTERMEDIATE, ane_channels=args.ane_channels,
        gpu_channels=INTERMEDIATE-args.ane_channels, activation="silu-gated", compute_dtype="fp16",
        source_operand_recipe=OPERANDS[args.operands], variant="fp16",
        compute_units="cpuAndNeuralEngine", artifacts={})
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key, value in configuration.items():
            if key != "artifacts" and previous.get(key) != value:
                raise ValueError(f"existing manifest differs: {key}")
        configuration = previous
    for block in blocks:
        key = str(block)
        if key in configuration["artifacts"]:
            artifact = configuration["artifacts"][key]
            if tree_identity(args.output / artifact["path"]) != artifact["identity"]:
                raise ValueError("existing compiled artifact identity changed")
            print(f"verified existing block {block}", flush=True)
            continue
        branch = "vid" if block < 10 else "all"
        prefix = f"blocks.{block}.mlp.{branch}."
        def matrix(name, shape):
            info = tensor_info(header, prefix+name+".weight", "F16", shape)
            return mapped_tensor(checkpoint, offset, info, np.float16, shape)
        gate = prepare_operand(matrix("proj_in_gate", (INTERMEDIATE, HIDDEN))[:args.ane_channels], args.operands)
        up = prepare_operand(matrix("proj_in", (INTERMEDIATE, HIDDEN))[:args.ane_channels], args.operands)
        down = prepare_operand(matrix("proj_out", (HIDDEN, INTERMEDIATE))[:, :args.ane_channels], args.operands)
        package = args.output / f"block-{block:02}.mlpackage"
        compiled = args.output / f"block-{block:02}.mlmodelc"
        if package.exists() or compiled.exists():
            raise ValueError(f"unregistered artifact exists: block {block}; inspect before re-exporting")
        print(f"exporting block {block}: {args.rows} rows, prefix {args.ane_channels}/{INTERMEDIATE}", flush=True)
        model = build_model(args.rows, args.ane_channels, gate, up, down, configuration["source_operand_recipe"])
        model.save(str(package))
        del model, gate, up, down
        loaded = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
        source_compiled = Path(loaded.get_compiled_model_path())
        shutil.copytree(source_compiled, compiled)
        loaded = None
        configuration["artifacts"][key] = dict(path=compiled.name, identity=tree_identity(compiled))
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(configuration, indent=2) + "\n")
        temporary.replace(manifest_path)
        print(f"compiled block {block}", flush=True)
    return configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--ane-channels", type=int, default=2048)
    parser.add_argument("--blocks", default="0-31")
    parser.add_argument("--operands", choices=tuple(OPERANDS), default="original-fp16")
    args = parser.parse_args()
    if args.rows not in (128, 256, 512, 1024) or not 0 < args.ane_channels < INTERMEDIATE or args.ane_channels % 256:
        parser.error("rows must be 128/256/512/1024; channels must be a positive 256-aligned prefix below 6912")
    export(args)


if __name__ == "__main__":
    main()
