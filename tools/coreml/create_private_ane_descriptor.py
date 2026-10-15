"""Create a checkpoint-independent private ANE shape descriptor.

This writes one small JSON file. It does not import Core ML, compile a graph,
or read/copy model weights. Public Core ML cannot load this descriptor.
"""

import argparse
import json
from pathlib import Path


def descriptor(kind, rows, hidden, width, tile_k=2048, tile_n=1024, lora_inputs=False):
    if kind not in ("matmul", "swiglu", "gelu"):
        raise ValueError("unsupported runtime graph kind")
    if type(lora_inputs) is not bool or (lora_inputs and kind != "swiglu"):
        raise ValueError("runtime LoRA activation inputs require SwiGLU")
    for value in (rows, hidden, width, tile_k, tile_n):
        if type(value) is not int or not 1 <= value <= 32768:
            raise ValueError("dimensions and tiles must be integers in [1,32768]")
    return {"schema_version": 1, "descriptor_version": 1,
            "backend": "private_runtime_shape", "kind": kind, "rows": rows,
            "hidden": hidden, "width": width, "tile_k": tile_k, "tile_n": tile_n,
            "layout": "out_in", "biases": False, "lora_inputs": lora_inputs}


def write_descriptor(path, spec):
    # Validate before creating anything, and preserve files/symlinks already at
    # the destination. The exclusive file creation is also safe against races.
    expected = descriptor(spec["kind"], spec["rows"], spec["hidden"], spec["width"],
                          spec["tile_k"], spec["tile_n"], spec["lora_inputs"])
    if spec != expected:
        raise ValueError("private descriptor must contain only its shape ABI")
    data = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(data)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("matmul", "swiglu", "gelu"), default="swiglu")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--full-width", "--width", dest="width", type=int, required=True,
                        help="full model FFN width; private channel selection happens at runtime")
    parser.add_argument("--tile-k", type=int, default=2048)
    parser.add_argument("--tile-n", type=int, default=1024)
    parser.add_argument("--lora-inputs", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file")
    args = parser.parse_args()
    try:
        spec = descriptor(args.kind, args.rows, args.hidden, args.width, args.tile_k,
                          args.tile_n, args.lora_inputs)
        write_descriptor(args.output, spec)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(str(args.output))


if __name__ == "__main__":
    main()
