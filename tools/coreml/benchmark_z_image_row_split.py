"""Single-block, real-input GPU/ANE FFN split screen.

By default this tests full-width ANE image rows versus full-width GPU remaining
rows. --channel-split instead tests a fixed-row, partial-width ANE image+caption
model against the original BF16 GPU complement. --image-only-channel-split
tests the existing partial-width, 1024-image-row ANE graph with the GPU
caption FFN kept at full BF16 width. None exercises the native
GPU/ANE bridge or establishes physical ANE INT8 placement. Inputs and weights
are resident before timing, so the parallel result is a pre-bridge screen.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time


def validate_split_manifest(manifest, capture_shape, block, channel_split,
                            image_only=False, two_group_image_probe=False):
    """Fail closed before loading checkpoint weights or compiling a Core ML model."""
    identity = manifest.get("export_identity", {})
    shape = manifest.get("shape", {})
    if two_group_image_probe and not image_only:
        raise ValueError("two-group probe requires image-only channel split")
    blocks = identity.get("blocks")
    valid_blocks = (blocks in ([block], [3, 29]) and block in blocks and
                    identity.get("hidden_a8_channel_groups") == 2
                    if two_group_image_probe else
                    blocks == (list(range(32)) if image_only else [block]))
    if (identity.get("activation_precision") != "int8" or
            shape.get("K") != 3840 or shape.get("N") != 3840 or
            shape.get("mlp_width") != 10240 or
            not valid_blocks or
            not isinstance(shape.get("buckets"), list) or len(shape["buckets"]) != 1 or
            not isinstance(capture_shape, tuple) or len(capture_shape) != 2 or
            capture_shape[1] != 3840 or not 1056 <= capture_shape[0] <= 2048 or
            (capture_shape[0] - 1024) % 32):
        raise ValueError("expected a single-block, fixed-shape Z-Image W8A8 probe")
    rows = shape["buckets"][0]
    channels = shape.get("ane_mlp_end")
    if channel_split and image_only:
        raise ValueError("choose channel split or image-only channel split")
    if image_only:
        if (identity.get("row_split_probe") or identity.get("image_only_token_rows") != 1024 or
                rows != 1024 or not isinstance(channels, int) or
                not 0 < channels < 10240 or channels % 32):
            raise ValueError("image-only channel probe needs a marked 1024-row artifact "
                             "and a proper ANE channel prefix")
    elif channel_split:
        if (identity.get("row_split_probe") or identity.get("region_image_rows") != 1024 or
                rows != capture_shape[0] or not isinstance(channels, int) or
                not 0 < channels < 10240 or channels % 32):
            raise ValueError("channel probe needs 1024-image-row region, full captured rows, "
                             "and a proper ANE channel prefix")
    elif (identity.get("row_split_probe") is not True or channels != 10240 or
          rows not in (256, 384, 416, 512, 544, 768, 1024)):
        raise ValueError("expected a single-block, fixed-shape, full-width W8A8 row probe")
    return rows, channels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--channel-split", action="store_true",
                   help="screen ANE prefix/GPU suffix channels on every captured row")
    p.add_argument("--image-only-channel-split", action="store_true",
                   help="screen existing image-only ANE channels plus full BF16 GPU caption")
    p.add_argument("--two-group-image-probe", action="store_true",
                   help="accept only the experimental 2-group, block-18 or blocks-3/29 image-only artifacts")
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.rounds < 2 or a.warmups < 1 or not 0 <= a.block < 32:
        p.error("expected >=2 rounds, >=1 warmup and Z-Image block 0...31")

    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native"))
    from benchmark_z_image_w8_suffix import read_bf16_suffix
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_z_image import SafetensorsSource

    manifest = json.loads(a.manifest.read_text())
    shape = manifest.get("shape", {})
    capture = np.load(a.capture, allow_pickle=False)
    if capture.dtype != np.float16:
        p.error("expected real [1024+32...1024,3840] FP16 FFN capture")
    try:
        rows, channels = validate_split_manifest(manifest, capture.shape, a.block,
                                                  a.channel_split,
                                                  a.image_only_channel_split,
                                                  a.two_group_image_probe)
    except ValueError as error:
        p.error(str(error))
    full_rows = capture.shape[0]
    ane_input = np.ascontiguousarray(
        (capture if a.channel_split else capture[:rows]).T[None, :, None, :])
    gpu_input = mx.array(capture[None, :rows] if a.image_only_channel_split else
                         capture[None] if a.channel_split else
                         capture[None, rows:]).astype(mx.bfloat16)
    full_input = mx.array(capture[None]).astype(mx.bfloat16)
    prefix = (f"noise_refiner.{a.block}" if a.block < 2 else
              f"layers.{a.block - 2}") + ".feed_forward"
    with SafetensorsSource.from_model(a.model) as reader:
        full_weights = [mx.array(read_bf16_suffix(reader, prefix + f".{name}.weight",
                    dim, 0, down, np)).astype(mx.bfloat16)
                   for name, dim, down in (
                       ("w1", (10240, 3840), False),
                       ("w3", (10240, 3840), False),
                       ("w2", (3840, 10240), True))]
        weights = ([mx.array(read_bf16_suffix(reader, prefix + f".{name}.weight",
                    dim, channels, down, np)).astype(mx.bfloat16)
                   for name, dim, down in (
                       ("w1", (10240, 3840), False),
                       ("w3", (10240, 3840), False),
                       ("w2", (3840, 10240), True))]
                   if a.channel_split or a.image_only_channel_split else full_weights)

    def gpu_ffn(value, matrices):
        gate = mx.matmul(value, matrices[0].T)
        up = mx.matmul(value, matrices[1].T)
        return mx.matmul((mx.sigmoid(gate) * gate) * up, matrices[2].T)

    mx.eval(full_input, gpu_input, *full_weights, *weights)
    package = a.manifest.parent / manifest["artifacts"][str(a.block)]["int8_pc"]
    if not package.is_dir():
        p.error("Core ML package missing")
    with tempfile.TemporaryDirectory(prefix="z-image-row-probe-") as temporary:
        compiled = Path(ct.models.utils.compile_model(
            str(package), destination_path=str(Path(temporary) / "row.mlmodelc")))
        units = ct.ComputeUnit.CPU_AND_NE
        plan = MLComputePlan.load_from_path(str(compiled), compute_units=units)
        preferred = {}
        for function in plan.model_structure.program.functions.values():
            for op in function.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is not None and "conv" in op.operator_name:
                    preferred[op.operator_name] = type(usage.preferred_compute_device).__name__
        ane = ct.models.CompiledMLModel(str(compiled), compute_units=units)

        def execute(mode):
            start = time.perf_counter()
            if mode == "native":
                result = gpu_ffn(full_input, full_weights)
                mx.eval(result)
                return (time.perf_counter() - start) * 1000, result, None
            if mode in ("gpu", "parallel"):
                gpu = gpu_ffn(gpu_input, weights)
                if a.image_only_channel_split:
                    gpu = mx.concatenate(
                        (gpu, gpu_ffn(full_input[:, rows:], full_weights)), axis=1)
                mx.async_eval(gpu)
            else:
                gpu = None
            ane_output = ane.predict({"x": ane_input})["y"] if mode in ("ane", "parallel") else None
            if gpu is not None:
                mx.eval(gpu)
            return (time.perf_counter() - start) * 1000, gpu, ane_output

        for _ in range(a.warmups):
            for mode in ("native", "gpu", "ane", "parallel"):
                execute(mode)
        samples = {mode: [] for mode in ("native", "gpu", "ane", "parallel")}
        for _ in range(a.rounds):
            for mode in ("native", "gpu", "ane", "parallel", "parallel", "ane", "gpu", "native"):
                elapsed, _, _ = execute(mode)
                samples[mode].append(elapsed)
        _, full, _ = execute("native")
        _, remainder, _ = execute("gpu")
        _, _, partial = execute("ane")
        target = np.asarray(full[0].astype(mx.float32))
        partial = np.asarray(partial[0, :, 0, :].T, dtype=np.float32) * shape["output_scale"]
        gpu = np.asarray(remainder[0].astype(mx.float32))
        candidate = (np.concatenate((partial + gpu[:rows], gpu[rows:]), axis=0)
                     if a.image_only_channel_split else partial + gpu if a.channel_split else
                     np.concatenate((partial, gpu), axis=0))
        def rel_l2(reference, tested):
            return float(np.linalg.norm((tested - reference).astype(np.float64)) /
                         max(float(np.linalg.norm(reference.astype(np.float64))), 1e-12))

        report = {
            "scope": "single-block, ready input, resident BF16 GPU + Core ML CPU_AND_NE; "
                     "not native transport, device trace or image-level result",
            "block": a.block,
            "split_axis": ("image_only_channels" if a.image_only_channel_split else
                           "channels" if a.channel_split else "rows"),
            "capture_rows": full_rows,
            "ane_rows": full_rows if a.channel_split else rows,
            "gpu_rows": full_rows if a.channel_split or a.image_only_channel_split else full_rows - rows,
            "ane_channels": channels,
            "gpu_channels": 10240 - channels if a.channel_split or a.image_only_channel_split else 10240,
            "caption_gpu_channels": 10240 if a.image_only_channel_split else None,
            "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            "capture_sha256": hashlib.sha256(a.capture.read_bytes()).hexdigest(),
            "rounds": a.rounds,
            "median_milliseconds": {key: statistics.median(value)
                                    for key, value in samples.items()},
            "min_milliseconds": {key: min(value) for key, value in samples.items()},
            "joined_vs_bf16_relative_l2": rel_l2(target, candidate),
            "image_vs_bf16_relative_l2": rel_l2(target[:1024], candidate[:1024]),
            "caption_vs_bf16_relative_l2": rel_l2(target[1024:], candidate[1024:]),
            "planned_conv_devices": preferred,
        }
        print(json.dumps(report))
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
