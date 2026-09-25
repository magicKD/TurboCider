"""Prepare exact BF16 block-18 weights and a held-out captured FFN input.

This is a single-block measurement input, not an exported or installed model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def validate_capture(capture, np):
    if capture.dtype != np.float16 or capture.shape != (1536, 3840) or not np.isfinite(capture).all():
        raise ValueError("expected finite held-out FP16 [1536,3840] block-18 capture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh dedicated probe file")

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
    from export_z_image import SafetensorsSource
    from benchmark_z_image_w8_suffix import read_bf16_suffix

    capture = np.load(args.capture, allow_pickle=False)
    validate_capture(capture, np)
    prefix = "layers.16.feed_forward"
    with SafetensorsSource.from_model(args.model) as reader:
        weights = {
            key: mx.array(read_bf16_suffix(reader, prefix + f".{name}.weight", shape,
                                             0, False, np)).astype(mx.bfloat16)
            for key, name, shape in (
                ("gate", "w1", (10240, 3840)),
                ("up", "w3", (10240, 3840)),
                ("down", "w2", (3840, 10240)),
            )
        }
    tensors = {"input": mx.array(capture[None]), **weights}
    mx.eval(*tensors.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), tensors)
    print(json.dumps({"output": str(args.output.resolve()),
                      "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                      "bundle_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "block": 18, "shape": list(capture.shape)}))


if __name__ == "__main__":
    main()
