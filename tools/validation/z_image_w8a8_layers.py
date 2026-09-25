"""Rank held-out Z-Image partial-FFN A8 errors across exported blocks.

This is an FP32 simulation of the two activation Q/DQ boundaries. It does not
simulate compressed W8 weights, Core ML FP16 rounding, ANE placement, or an
end-to-end denoising trajectory. Use the separate Core ML branch validator and
matched image tests before making quality claims.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
from export_z_image import SafetensorsSource
from z_image_smoothquant import (load_calibration, load_calibration_union, smooth_partial_ffn,
                                 calibrate_input_a8, calibrate_hidden_a8,
                                 calibrate_hidden_channel_groups,
                                 calibrate_adaptive_hidden_a8,
                                 screen_hidden_range_order,
                                 screen_joint_hidden_channel_groups, representative_rows)


def swiglu(x, gate, up, np):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = x @ gate.T
        return (projected / (1 + np.exp(-np.clip(projected, -80, 80)))) * (x @ up.T)


def quantize_dequantize(x, scale, np):
    if not np.all((0 < scale) & (scale < float("inf"))):
        raise ValueError("invalid A8 scale")
    return np.clip(np.rint(x / scale), -127, 127) * scale


def dynamic_hidden_a8(hidden, down, np):
    """Diagnostic only: derive a max-based hidden A8 scale from each input row.

    This is NOT a deployable Core ML A8 op: MIL quantize requires a constant
    scale, while these scales depend on the current sample's activation.
    """
    if hidden.ndim != 2 or down.ndim != 2 or hidden.shape[1] != down.shape[1]:
        raise ValueError("invalid dynamic hidden A8 geometry")
    row_peak = np.max(np.abs(hidden), axis=1, keepdims=True)
    row_scale = np.maximum(row_peak / 127., 1e-6).astype(np.float16).astype(np.float32)
    return quantize_dequantize(hidden, row_scale, np) @ down.T


def adaptive_hidden_a8(hidden, down, scales, np, shared_qdq=False):
    """Diagnostic: select one of the training-fixed Q/DQ scales per image row."""
    if (hidden.ndim != 2 or down.ndim != 2 or hidden.shape[1] != down.shape[1] or
            np.ndim(scales) != 1 or len(scales) < 2 or
            not np.all(np.diff(scales) > 0)):
        raise ValueError("invalid adaptive hidden A8 geometry or scales")
    row_peak = np.max(np.abs(hidden), axis=1)
    indexes = np.searchsorted(scales * 127., row_peak, side="left")
    chosen = np.asarray(scales)[np.minimum(indexes, len(scales) - 1), None]
    if shared_qdq:
        # Mirror the Core ML FP16 pre/post multiplication and its constant
        # quantization scale. This is a different numerical path from Q/DQ at
        # the individually selected scale.
        base = np.float16(scales[-1])
        ratio = (base / chosen).astype(np.float16)
        inverse = (chosen / base).astype(np.float16)
        prepared = (hidden.astype(np.float16) * ratio).astype(np.float16)
        quantized = quantize_dequantize(prepared, float(base), np).astype(np.float16)
        return (quantized * inverse).astype(np.float32) @ down.T
    return quantize_dequantize(hidden, chosen, np) @ down.T


def calibrate_static_hidden_row_groups(gate, up, down, s1, samples, input_scale,
                                       activation_scale, image_rows, groups, np,
                                       rows_per_group=4, output_aware=False):
    """Training-only, max-based hidden scales for fixed image-row positions.

    This is a diagnostic, not an exporter: static grouped scales are plausible
    MIL vector quantize inputs, but device placement and image quality are
    unknown. Match the reference A8 first boundary on sampled training rows.
    """
    if (groups < 2 or image_rows < groups or image_rows % groups or
            rows_per_group < 1 or not samples or gate.shape != up.shape or
            gate.shape[1] != len(s1) or down.shape[1] != gate.shape[0] or
            activation_scale <= 0):
        raise ValueError("invalid static hidden row-group calibration")
    width = image_rows // groups
    sample_count = min(rows_per_group, width)
    positions = np.concatenate([
        np.linspace(i * width, (i + 1) * width - 1, sample_count,
                    dtype=np.int64)
        for i in range(groups)])
    maxima = np.zeros(groups, dtype=np.float32)
    hidden_groups = [[] for _ in range(groups)] if output_aware else None
    for sample in samples:
        if sample.ndim != 2 or len(sample) < image_rows or sample.shape[1] != len(s1):
            raise ValueError("invalid static hidden row-group training sample")
        x = quantize_dequantize(sample[positions].astype(np.float32) / s1,
                                input_scale, np)
        hidden = swiglu(x, gate, up, np) / activation_scale ** 2
        if not np.isfinite(hidden).all():
            raise ValueError("nonfinite hidden row-group calibration")
        maxima = np.maximum(maxima, np.max(np.abs(hidden.reshape(
            groups, sample_count, -1)), axis=(1, 2)))
        if output_aware:
            for index, chunk in enumerate(hidden.reshape(groups, sample_count, -1)):
                hidden_groups[index].append(chunk)
    scales = np.maximum(maxima / 127., 1e-6).astype(np.float16).astype(np.float32)
    if output_aware:
        for index, chunks in enumerate(hidden_groups):
            hidden = np.concatenate(chunks)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                target = hidden @ down.T
            if not np.isfinite(target).all():
                raise ValueError("nonfinite static row-group partial FFN training target")
            trials = []
            for percentile in (99., 99.5, 99.9, 99.95, 99.99, 99.999, 100.):
                threshold = np.percentile(np.abs(hidden), percentile)
                scale = np.float16(max(float(threshold / 127.), 1e-6))
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    projected = quantize_dequantize(hidden, float(scale), np) @ down.T
                if not np.isfinite(projected).all():
                    continue
                score = float(np.sum((projected - target).astype(np.float64) ** 2))
                trials.append((score, float(scale)))
            if not trials:
                raise ValueError("no finite static row-group A8 threshold")
            scales[index] = min(trials)[1]
    if not np.isfinite(scales).all():
        raise ValueError("nonfinite hidden row-group A8 scale")
    return scales


def measure_a8(x, gate, up, down, s1, input_scale, hidden_scale, activation_scale, np,
               channel_scales=None, input_channel_scales=None, dynamic_hidden=False,
               static_row_scales=None, adaptive_scales=None, adaptive_shared=False):
    """Attribute input and hidden A8 damage against the same FP32 reference."""
    x = np.asarray(x, dtype=np.float32)
    normalized = x / s1
    if input_channel_scales is not None:
        if len(input_channel_scales) < 2 or normalized.shape[1] % len(input_channel_scales):
            raise ValueError("invalid input A8 channel-group scales")
        input_width = normalized.shape[1] // len(input_channel_scales)
        input_a8 = np.concatenate([
            quantize_dequantize(normalized[:, i * input_width:(i + 1) * input_width],
                                scale, np)
            for i, scale in enumerate(input_channel_scales)], axis=1)
    else:
        input_a8 = quantize_dequantize(normalized, input_scale, np)
    original_hidden = swiglu(normalized, gate, up, np) / (activation_scale ** 2)
    input_hidden = swiglu(input_a8, gate, up, np) / (activation_scale ** 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        reference = original_hidden @ down.T
        input_only = input_hidden @ down.T
        hidden_only = quantize_dequantize(original_hidden, hidden_scale, np) @ down.T
        combined = quantize_dequantize(input_hidden, hidden_scale, np) @ down.T
        dynamic = dynamic_hidden_a8(input_hidden, down, np) if dynamic_hidden else None
        static_rows = (quantize_dequantize(input_hidden, static_row_scales, np) @ down.T
                       if static_row_scales is not None else None)
        adaptive = (adaptive_hidden_a8(input_hidden, down, adaptive_scales, np)
                    if adaptive_scales is not None else None)
        shared = (adaptive_hidden_a8(input_hidden, down, adaptive_scales, np,
                                     shared_qdq=True)
                  if adaptive_scales is not None and adaptive_shared else None)
        grouped = None
        if channel_scales is not None:
            if len(channel_scales) < 2 or original_hidden.shape[1] % len(channel_scales):
                raise ValueError("invalid hidden A8 channel-group scales")
            width = original_hidden.shape[1] // len(channel_scales)
            def grouped_qdq(hidden):
                return np.concatenate([
                    quantize_dequantize(hidden[:, i * width:(i + 1) * width],
                                        scale, np)
                    for i, scale in enumerate(channel_scales)], axis=1)
            grouped = grouped_qdq(input_hidden) @ down.T
    if not all(np.isfinite(v).all() for v in
               (reference, input_only, hidden_only, combined, grouped,
                dynamic, static_rows, adaptive, shared)
               if v is not None):
        raise ValueError("nonfinite partial FFN diagnostic")
    reference_energy = float(np.sum(reference.astype(np.float64) ** 2))
    def nmse(candidate):
        return float(np.sum((candidate - reference).astype(np.float64) ** 2) /
                     max(reference_energy, 1e-12))
    return {
        "reference_energy": reference_energy,
        "input_a8_squared_error": nmse(input_only) * reference_energy,
        "hidden_a8_squared_error": nmse(hidden_only) * reference_energy,
        "combined_a8_squared_error": nmse(combined) * reference_energy,
        "input_clip_fraction": float(np.mean(np.abs(normalized) > 127 * input_scale)),
        "hidden_clip_fraction": float(np.mean(np.abs(input_hidden) > 127 * hidden_scale)),
        **({"channel_group_a8_squared_error": nmse(grouped) * reference_energy}
           if grouped is not None else {}),
        **({"dynamic_hidden_a8_squared_error": nmse(dynamic) * reference_energy}
           if dynamic is not None else {}),
        **({"static_row_hidden_a8_squared_error": nmse(static_rows) * reference_energy}
           if static_rows is not None else {}),
        **({"adaptive_hidden_a8_squared_error": nmse(adaptive) * reference_energy}
           if adaptive is not None else {}),
        **({"adaptive_shared_a8_squared_error": nmse(shared) * reference_energy}
           if shared is not None else {}),
    }


def select_rows(sample, count, np):
    if sample.ndim != 2 or count < 1:
        raise ValueError("invalid held-out sample or row count")
    return np.asarray(sample[representative_rows(sample, count, np)],
                      dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--extra-calibration-dir", type=Path, action="append", default=[],
                        help="additional training directories in exporter order; never held-out data")
    parser.add_argument("--allow-unexported-calibration", action="store_true",
                        help="rederive image A8 scales from this calibration directory; "
                             "simulation only, never validates the exported artifact")
    parser.add_argument("--heldout-dir", required=True, type=Path)
    parser.add_argument("--blocks", default="all", help="all or comma-separated block indexes")
    parser.add_argument("--samples-per-block", type=int, default=2)
    parser.add_argument("--rows-per-sample", type=int, default=16)
    parser.add_argument("--full-ffn-norm", action="store_true",
                        help="also normalize partial A8 error against the entire BF16 FFN output")
    parser.add_argument("--candidate-input-a8", action="store_true",
                        help="evaluate a training-calibrated output-aware input A8 threshold")
    parser.add_argument("--image-rows", type=int,
                        help="evaluate the image-token region of a region-split A8 manifest")
    parser.add_argument("--image-tiles", type=int,
                        help="simulate separately calibrated hidden A8 for equal image-row tiles")
    parser.add_argument("--hidden-channel-groups", type=int,
                        help="simulate separate hidden A8 scales across channel groups")
    parser.add_argument("--input-channel-groups", type=int,
                        help="simulate image-token input A8 scales per input-channel group")
    parser.add_argument("--sort-hidden-channels", action="store_true",
                        help="training-only hidden-range order before grouped A8; simulation only")
    parser.add_argument("--diagnose-dynamic-hidden-a8", action="store_true",
                        help="held-out per-token max-based hidden A8; non-deployable diagnostic")
    parser.add_argument("--diagnose-static-hidden-row-groups", type=int,
                        help="train max-based hidden A8 per fixed image-row group; simulation only")
    parser.add_argument("--static-hidden-output-aware", action="store_true",
                        help="search static row-group scales by training partial-output error")
    parser.add_argument("--diagnose-adaptive-hidden-a8-bins", type=int,
                        help="training-fixed scale bank selected per held-out image row; simulation only")
    parser.add_argument("--diagnose-adaptive-shared-qdq", action="store_true",
                        help="simulate FP16 pre/post scaling around one constant hidden Q/DQ")
    parser.add_argument("--joint-hidden-channel-groups", type=int,
                        help="screen jointly optimized hidden A8 scales (calibration only)")
    args = parser.parse_args()
    import numpy as np

    manifest = json.loads(args.manifest.read_text())
    identity = manifest["export_identity"]
    shape = manifest["shape"]
    if (identity.get("activation_precision") != "int8" or
            identity.get("checkpoint_format") != "safetensors" or
            shape["ane_mlp_start"] != 0 or len(shape["buckets"]) != 1 or
            any(path.is_symlink() or not path.is_dir() for path in
                [args.calibration_dir, *args.extra_calibration_dir]) or
            args.heldout_dir.is_symlink() or not args.heldout_dir.is_dir() or
            args.samples_per_block < 1 or args.rows_per_sample < 1):
        raise ValueError("requires a fixed-bucket, safetensors W8A8 manifest and real samples")
    calibration_dirs = [args.calibration_dir, *args.extra_calibration_dir]
    if any(path.resolve() == args.heldout_dir.resolve() for path in calibration_dirs):
        raise ValueError("held-out samples must be independent of calibration")
    if not args.allow_unexported_calibration and len(calibration_dirs) != identity.get(
            "calibration_directory_count", 1):
        raise ValueError("calibration directory count differs from export")
    width, hidden, rows = shape["ane_mlp_end"], shape["K"], shape["buckets"][0]
    if args.image_rows is not None and args.image_rows != identity.get("region_image_rows"):
        raise ValueError("image-rows must match the region-split export identity")
    if args.image_tiles is not None and (args.image_rows is None or args.image_tiles < 2 or
            args.image_rows % args.image_tiles):
        raise ValueError("image-tiles requires an evenly divided image region")
    if args.hidden_channel_groups is not None and (args.image_rows is None or
            args.hidden_channel_groups < 2 or width % args.hidden_channel_groups):
        raise ValueError("hidden-channel-groups requires a divisible image-region width")
    if args.input_channel_groups is not None and (args.image_rows is None or
            args.input_channel_groups < 2 or hidden % args.input_channel_groups or
            hidden // args.input_channel_groups % 32):
        raise ValueError("input-channel-groups requires a 32-aligned image-region width")
    if args.sort_hidden_channels and args.hidden_channel_groups is None:
        raise ValueError("hidden channel sorting requires hidden-channel-groups")
    if args.joint_hidden_channel_groups is not None and (args.image_rows is None or
            args.joint_hidden_channel_groups < 2 or width % args.joint_hidden_channel_groups):
        raise ValueError("joint-hidden-channel-groups requires a divisible image-region width")
    if args.allow_unexported_calibration and args.image_rows is None:
        raise ValueError("unexported calibration screen requires image-rows")
    if args.diagnose_static_hidden_row_groups is not None and (
            args.image_rows is None or args.diagnose_static_hidden_row_groups < 2 or
            args.image_rows % args.diagnose_static_hidden_row_groups):
        raise ValueError("static hidden row groups require a divisible image region")
    if args.static_hidden_output_aware and args.diagnose_static_hidden_row_groups is None:
        raise ValueError("output-aware threshold search requires static hidden row groups")
    if args.diagnose_adaptive_hidden_a8_bins is not None and (
            args.image_rows is None or args.diagnose_adaptive_hidden_a8_bins < 2):
        raise ValueError("adaptive hidden A8 bins require image rows and at least two scales")
    if args.diagnose_adaptive_shared_qdq and args.diagnose_adaptive_hidden_a8_bins is None:
        raise ValueError("adaptive shared Q/DQ requires a scale bank")
    blocks = sorted(map(int, manifest["artifacts"])) if args.blocks == "all" else [
        int(part) for part in args.blocks.split(",")]
    if len(set(blocks)) != len(blocks) or any(str(block) not in manifest["artifacts"] for block in blocks):
        raise ValueError("blocks must be unique exported block indexes")
    with SafetensorsSource.from_model(args.model) as reader:
        if str(reader.checkpoint) != identity["checkpoint"]:
            raise ValueError("model checkpoint does not match export identity")
        results = []
        for block in blocks:
            calibration, digest = load_calibration_union(
                [path / f"block{block}" for path in calibration_dirs],
                rows, hidden, np, pad_rows=block < 2)
            if not args.allow_unexported_calibration and digest != identity["calibration"][str(block)]["sha256"]:
                raise ValueError(f"block {block} calibration differs from export")
            heldout, _ = load_calibration(args.heldout_dir / f"block{block}", rows, hidden,
                                          np, pad_rows=block < 2)
            prefix = (f"noise_refiner.{block}" if block < 2 else
                      f"layers.{block - 2}") + ".feed_forward"
            full_gate = reader.tensor(prefix + ".w1.weight", (10240, hidden), np)
            full_up = reader.tensor(prefix + ".w3.weight", (10240, hidden), np)
            full_down = reader.tensor(prefix + ".w2.weight", (hidden, 10240), np)
            gate, up, down, s1, _ = smooth_partial_ffn(
                full_gate[:width], full_up[:width], full_down[:, :width], calibration,
                identity["sq_alpha1"], identity["sq_alpha2"], np, identity["sq_hidden_rows"],
                outlier_rows=("outlier_rows_v5" in identity["a8_graph"] or
                              "regions" in identity["a8_graph"]),
                hidden_region_rows=(identity.get("region_image_rows")
                                    if "regions_image_sq" in identity["a8_graph"]
                                    else None))
            scales = manifest["activation_quantization"][str(block)]
            if args.allow_unexported_calibration:
                # Mirror the exporter's maximum-based input scale and
                # output-aware hidden threshold. The new SQ weights and
                # scales have NOT been exported to Core ML.
                image_samples = [sample[:args.image_rows] for sample in calibration]
                image_input = np.float16(max(
                    max(float(np.max(np.abs(sample.astype(np.float32) / s1)))
                        for sample in image_samples) / 127, 1e-6))
                image_hidden = calibrate_hidden_a8(
                    gate, up, down, s1, image_samples, float(image_input),
                    shape["activation_scale"], np, outlier_rows=True)[0]
                scales = {"region_input_a8_scales": [float(image_input)],
                          "region_hidden_a8_scales": [float(image_hidden)]}
            tile_scales = None
            if args.image_tiles is not None:
                tile_rows = args.image_rows // args.image_tiles
                tile_scales = [float(calibrate_hidden_a8(
                    gate, up, down, s1,
                    [sample[tile * tile_rows:(tile + 1) * tile_rows]
                     for sample in calibration],
                    scales["region_input_a8_scales"][0], shape["activation_scale"], np,
                    outlier_rows=True)[0]) for tile in range(args.image_tiles)]
            channel_scales = (calibrate_hidden_channel_groups(
                gate, up, down, s1, [sample[:args.image_rows] for sample in calibration],
                scales["region_input_a8_scales"][0], shape["activation_scale"],
                args.hidden_channel_groups, np)
                if args.hidden_channel_groups is not None else None)
            sorted_order, sorted_scales = None, None
            if args.sort_hidden_channels:
                sorted_order = screen_hidden_range_order(
                    gate, up, s1, [sample[:args.image_rows] for sample in calibration],
                    scales["region_input_a8_scales"][0], shape["activation_scale"], np)
                sorted_scales = calibrate_hidden_channel_groups(
                    gate[sorted_order], up[sorted_order], down[:, sorted_order], s1,
                    [sample[:args.image_rows] for sample in calibration],
                    scales["region_input_a8_scales"][0], shape["activation_scale"],
                    args.hidden_channel_groups, np)
            input_channel_scales = None
            if args.input_channel_groups is not None:
                step = hidden // args.input_channel_groups
                input_channel_scales = [float(np.float16(max(max(
                    float(np.max(np.abs(sample[:args.image_rows, begin:begin + step].astype(
                        np.float32) / s1[begin:begin + step])))
                    for sample in calibration) / 127, 1e-6)))
                    for begin in range(0, hidden, step)]
            joint_scales, joint_train = (screen_joint_hidden_channel_groups(
                gate, up, down, s1, [sample[:args.image_rows] for sample in calibration],
                scales["region_input_a8_scales"][0], shape["activation_scale"],
                args.joint_hidden_channel_groups, np)
                if args.joint_hidden_channel_groups is not None else (None, None))
            candidate_samples = ([sample[:args.image_rows] for sample in calibration]
                                 if args.image_rows is not None else calibration)
            static_row_scales = (calibrate_static_hidden_row_groups(
                gate, up, down, s1, candidate_samples,
                scales["region_input_a8_scales"][0], shape["activation_scale"],
                args.image_rows, args.diagnose_static_hidden_row_groups, np,
                output_aware=args.static_hidden_output_aware)
                if args.diagnose_static_hidden_row_groups is not None else None)
            adaptive_scales = (calibrate_adaptive_hidden_a8(
                gate, up, s1, candidate_samples,
                scales["region_input_a8_scales"][0], shape["activation_scale"],
                args.image_rows, args.diagnose_adaptive_hidden_a8_bins, np)
                if args.diagnose_adaptive_hidden_a8_bins is not None else None)
            candidate = (calibrate_input_a8(gate, up, down, s1, candidate_samples, np,
                                            outlier_rows=True)[0]
                         if args.candidate_input_a8 else None)
            totals = {key: 0. for key in ("reference_energy", "input_a8_squared_error",
                                            "hidden_a8_squared_error", "combined_a8_squared_error",
                                            "input_clip_fraction", "hidden_clip_fraction")}
            samples = heldout[:args.samples_per_block]
            full_energy = 0.
            candidate_error = candidate_clipping = 0.
            tile_old_error = tile_new_error = tile_energy = 0.
            grouped_error = 0.
            dynamic_error = 0.
            static_row_error = 0.
            adaptive_error = 0.
            shared_error = 0.
            joint_error = 0.
            input_grouped_error = 0.
            sorted_grouped_error = 0.
            for sample in samples:
                region = sample[:args.image_rows] if args.image_rows is not None else sample
                selected_rows = representative_rows(region, args.rows_per_sample, np)
                x = np.asarray(region[selected_rows], dtype=np.float32)
                input_scale = (scales["region_input_a8_scales"][0]
                               if args.image_rows is not None else scales["input_a8_scale"])
                hidden_scale = (scales["region_hidden_a8_scales"][0]
                                if args.image_rows is not None else scales["hidden_a8_scale"])
                metrics = measure_a8(x, gate, up,
                                     down, s1, input_scale, hidden_scale,
                                     shape["activation_scale"], np,
                                     channel_scales=channel_scales,
                                     dynamic_hidden=args.diagnose_dynamic_hidden_a8,
                                     static_row_scales=(static_row_scales[
                                         selected_rows // (args.image_rows // len(static_row_scales)), None]
                                         if static_row_scales is not None else None),
                                     adaptive_scales=adaptive_scales,
                                     adaptive_shared=args.diagnose_adaptive_shared_qdq)
                if args.diagnose_dynamic_hidden_a8:
                    dynamic_error += metrics["dynamic_hidden_a8_squared_error"]
                if static_row_scales is not None:
                    static_row_error += metrics["static_row_hidden_a8_squared_error"]
                if adaptive_scales is not None:
                    adaptive_error += metrics["adaptive_hidden_a8_squared_error"]
                if args.diagnose_adaptive_shared_qdq:
                    shared_error += metrics["adaptive_shared_a8_squared_error"]
                if channel_scales is not None:
                    grouped_error += metrics["channel_group_a8_squared_error"]
                if joint_scales is not None:
                    joint_error += measure_a8(x, gate, up, down, s1, input_scale,
                                              hidden_scale, shape["activation_scale"], np,
                                              channel_scales=joint_scales)[
                        "channel_group_a8_squared_error"]
                if input_channel_scales is not None:
                    input_grouped_error += measure_a8(
                        x, gate, up, down, s1, input_scale, hidden_scale,
                        shape["activation_scale"], np,
                        channel_scales=channel_scales,
                        input_channel_scales=input_channel_scales)[
                            "channel_group_a8_squared_error" if channel_scales is not None
                            else "combined_a8_squared_error"]
                if sorted_order is not None:
                    sorted_grouped_error += measure_a8(
                        x, gate[sorted_order], up[sorted_order], down[:, sorted_order],
                        s1, input_scale, hidden_scale, shape["activation_scale"], np,
                        channel_scales=sorted_scales)["channel_group_a8_squared_error"]
                for key, value in metrics.items():
                    if key in totals:
                        totals[key] += value
                if tile_scales is not None:
                    for tile in range(args.image_tiles):
                        tile_x = select_rows(sample[tile * tile_rows:(tile + 1) * tile_rows],
                                             args.rows_per_sample, np)
                        old = measure_a8(tile_x, gate, up, down, s1, input_scale,
                                         hidden_scale, shape["activation_scale"], np)
                        new = measure_a8(tile_x, gate, up, down, s1, input_scale,
                                         tile_scales[tile], shape["activation_scale"], np)
                        tile_energy += old["reference_energy"]
                        tile_old_error += old["combined_a8_squared_error"]
                        tile_new_error += new["combined_a8_squared_error"]
                if candidate is not None:
                    proposed = measure_a8(x, gate, up, down, s1, float(candidate),
                                          hidden_scale, shape["activation_scale"], np)
                    candidate_error += proposed["combined_a8_squared_error"]
                    candidate_clipping += proposed["input_clip_fraction"]
                if args.full_ffn_norm:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        entire = swiglu(x, full_gate.astype(np.float32),
                                        full_up.astype(np.float32), np) @ full_down.astype(np.float32).T
                    if not np.isfinite(entire).all():
                        raise ValueError(f"block {block} full FFN is nonfinite")
                    full_energy += float(np.sum(entire.astype(np.float64) ** 2))
            energy = max(totals["reference_energy"], 1e-12)
            result = {"block": block, "samples": len(samples),
                      "calibration_sha256": digest,
                      "unexported_calibration_screen": args.allow_unexported_calibration,
                      "rows_per_sample": min(args.rows_per_sample, args.image_rows or rows),
                      **({"region": "image"} if args.image_rows is not None else {}),
                      "reference_energy": totals["reference_energy"],
                      **{key.replace("_squared_error", "_relative_l2"):
                         (totals[key] / energy) ** 0.5
                         for key in ("input_a8_squared_error", "hidden_a8_squared_error",
                                     "combined_a8_squared_error")},
                      "input_clip_fraction": totals["input_clip_fraction"] / len(samples),
                      "hidden_clip_fraction": totals["hidden_clip_fraction"] / len(samples)}
            if args.full_ffn_norm:
                result["full_ffn_energy"] = full_energy
                result["combined_a8_vs_full_relative_l2"] = (
                    totals["combined_a8_squared_error"] * shape["activation_scale"] ** 4 /
                    max(full_energy, 1e-12)) ** 0.5
            if candidate is not None:
                result["candidate_input_a8_scale"] = float(candidate)
                result["candidate_combined_relative_l2"] = (candidate_error / energy) ** 0.5
                result["candidate_input_clip_fraction"] = candidate_clipping / len(samples)
            if tile_scales is not None:
                result["image_tile_hidden_a8_scales"] = tile_scales
                result["tile_sampled_global_relative_l2"] = (tile_old_error / tile_energy) ** 0.5
                result["tile_sampled_candidate_relative_l2"] = (tile_new_error / tile_energy) ** 0.5
            if channel_scales is not None:
                result["channel_group_hidden_a8_scales"] = list(map(float, channel_scales))
                result["channel_group_candidate_relative_l2"] = (grouped_error / energy) ** 0.5
            if args.diagnose_dynamic_hidden_a8:
                result["dynamic_hidden_a8_diagnostic_relative_l2"] = (
                    dynamic_error / energy) ** .5
            if static_row_scales is not None:
                result["static_row_hidden_a8_diagnostic_relative_l2"] = (
                    static_row_error / energy) ** .5
                result["static_row_hidden_a8_output_aware"] = args.static_hidden_output_aware
                result["static_row_hidden_a8_scale_range"] = [
                    float(static_row_scales.min()), float(static_row_scales.max())]
            if adaptive_scales is not None:
                result["adaptive_hidden_a8_diagnostic_relative_l2"] = (
                    adaptive_error / energy) ** .5
                result["adaptive_hidden_a8_training_scales"] = list(map(float, adaptive_scales))
                if args.diagnose_adaptive_shared_qdq:
                    result["adaptive_shared_a8_diagnostic_relative_l2"] = (
                        shared_error / energy) ** .5
            if joint_scales is not None:
                result["joint_group_hidden_a8_scales"] = list(map(float, joint_scales))
                result["joint_group_train"] = joint_train
                result["joint_group_heldout_relative_l2"] = (joint_error / energy) ** 0.5
            if input_channel_scales is not None:
                result["input_channel_group_scales"] = input_channel_scales
                result["input_channel_group_candidate_relative_l2"] = (
                    input_grouped_error / energy) ** .5
            if sorted_order is not None:
                result["sorted_hidden_order_sha256"] = hashlib.sha256(
                    sorted_order.tobytes()).hexdigest()
                result["sorted_hidden_channel_scales"] = list(map(float, sorted_scales))
                result["sorted_hidden_group_candidate_relative_l2"] = (
                    sorted_grouped_error / energy) ** .5
            results.append(result)
            print(json.dumps(result), flush=True)
    ranking = ("combined_a8_vs_full_relative_l2" if args.full_ffn_norm else
               "combined_a8_relative_l2")
    summary = {"ranked_by": ranking, "worst_combined_blocks": [item["block"] for item in sorted(
        results, key=lambda item: item[ranking], reverse=True)[:10]],
        "scope": "heldout_partial_ffn_fp32_activation_qdq_only; not W8, Core ML, ANE or image quality"}
    if args.candidate_input_a8:
        summary["candidate_improved_blocks"] = [item["block"] for item in results if
            item["candidate_combined_relative_l2"] < item["combined_a8_relative_l2"]]
        summary["candidate_regressed_blocks"] = [item["block"] for item in results if
            item["candidate_combined_relative_l2"] > item["combined_a8_relative_l2"]]
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
