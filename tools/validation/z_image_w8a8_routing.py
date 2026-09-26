"""Screen output-aware ANE hidden-channel routing on held-out Z-Image inputs.

This FP32 A8 simulation omits W8, Core ML rounding, ANE execution, and GPU
W8 error. Groups are selected *only* on calibration samples; held-out error
is compared against the original contiguous ANE prefix. An optional caption
term balances image/caption NMSE during selection; its default is image-only
for compatibility with existing routes. It does not modify a checkpoint or
claim image fidelity.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
from export_z_image import SafetensorsSource
from z_image_smoothquant import (load_calibration, representative_rows,
                                 smooth_partial_ffn, calibrate_hidden_a8)


def swiglu(value, gate, up, np):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g = value @ gate.T
        return (g / (1 + np.exp(-np.clip(g, -80, 80)))) * (value @ up.T)


def qdq(value, scale, np):
    return np.clip(np.rint(value / float(scale)), -127, 127) * float(scale)


def select_image_rows(samples, count, image_rows, np):
    return np.concatenate([sample[representative_rows(
        sample[:image_rows], count, np)].astype(np.float32)
        for sample in samples], axis=0)


def route_groups(errors, count, np):
    """Greedily minimize the combined error, including error cancellation."""
    if (errors.ndim != 3 or count < 1 or count > len(errors) or
            not np.isfinite(errors).all()):
        raise ValueError("invalid routing error matrix or group count")
    accumulated = np.zeros_like(errors[0])
    available = set(range(len(errors)))
    selected = []
    for _ in range(count):
        best = min(available, key=lambda index: float(np.sum(
            (accumulated + errors[index]).astype(np.float64) ** 2)))
        selected.append(best)
        accumulated += errors[best]
        available.remove(best)
    return sorted(selected)


def combine_region_errors(image_errors, caption_errors, image_reference,
                          caption_reference, caption_weight, np):
    """Give caption a specified weight in relative-output-error space.

    Both errors are in the same output units. The raw caption energy can be
    orders of magnitude larger than image energy, so use region-specific
    reference norms rather than allowing one region to silently dominate.
    """
    if (image_errors.ndim != 3 or caption_errors.ndim != 3 or
            image_errors.shape[0] != caption_errors.shape[0] or
            image_errors.shape[2] != caption_errors.shape[2] or
            image_reference.shape != image_errors.shape[1:] or
            caption_reference.shape != caption_errors.shape[1:] or
            not np.isfinite(image_errors).all() or
            not np.isfinite(caption_errors).all() or
            not np.isfinite(image_reference).all() or
            not np.isfinite(caption_reference).all() or
            not np.isfinite(caption_weight) or caption_weight < 0):
        raise ValueError("invalid caption-aware routing errors or reference")
    if caption_weight == 0:
        return image_errors
    image_energy = float(np.sum(image_reference.astype(np.float64) ** 2))
    caption_energy = float(np.sum(caption_reference.astype(np.float64) ** 2))
    if image_energy <= 0 or caption_energy <= 0:
        raise ValueError("caption-aware routing requires nonzero region reference energy")
    factor = (caption_weight * image_energy / caption_energy) ** .5
    return np.concatenate((image_errors, caption_errors * factor), axis=1)


def constrained_caption_route(image_errors, caption_errors, image_route,
                              image_slack, np):
    """Swap training groups for caption gain without exceeding an image SSE cap.

    The baseline is the existing image-only route, not the contiguous prefix.
    A Gram matrix makes candidate swaps cheap without materializing a new
    [sample, hidden] tensor for every pair. Both regions remain training-only.
    """
    if (image_errors.ndim != 3 or caption_errors.ndim != 3 or
            image_errors.shape[0] != caption_errors.shape[0] or
            image_errors.shape[2] != caption_errors.shape[2] or
            not image_route or len(image_route) >= len(image_errors) or
            len(set(image_route)) != len(image_route) or
            any(index < 0 or index >= len(image_errors) for index in image_route) or
            not np.isfinite(image_errors).all() or not np.isfinite(caption_errors).all() or
            not np.isfinite(image_slack) or image_slack < 0):
        raise ValueError("invalid constrained caption routing inputs")

    def gram(errors):
        flat = errors.reshape(len(errors), -1)
        value = np.zeros((len(errors), len(errors)), np.float64)
        for first in range(0, flat.shape[1], 65536):
            chunk = flat[:, first:first + 65536].astype(np.float64)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                value += chunk @ chunk.T
        if not np.isfinite(value).all():
            raise ValueError("nonfinite constrained caption routing Gram matrix")
        return value

    image_gram, caption_gram = gram(image_errors), gram(caption_errors)

    def energy(matrix, selected):
        return float(np.sum(matrix[np.ix_(selected, selected)]))

    selected = sorted(image_route)
    base_image = energy(image_gram, selected)
    ceiling = base_image * (1 + image_slack) + 1e-12
    caption_score = energy(caption_gram, selected)
    swaps = []
    for _ in range(len(selected)):
        best = None
        for old in selected:
            for new in range(len(image_errors)):
                if new in selected:
                    continue
                trial = sorted((set(selected) - {old}) | {new})
                image_score = energy(image_gram, trial)
                if image_score > ceiling:
                    continue
                trial_caption = energy(caption_gram, trial)
                if trial_caption < caption_score - max(caption_score * 1e-8, 1e-12):
                    candidate = (trial_caption, image_score, old, new, trial)
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
        if best is None:
            break
        caption_score, image_score, old, new, selected = best
        swaps.append({"old": old, "new": new, "image_sse_vs_initial": image_score / max(base_image, 1e-30)})
    return selected, swaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--group-width", type=int, default=256)
    parser.add_argument("--ane-width", type=int,
                        help="candidate ANE width; source manifest supplies calibration provenance")
    parser.add_argument("--calibration-rows", type=int, default=32)
    parser.add_argument("--heldout-rows", type=int, default=32)
    parser.add_argument("--caption-route-weight", type=float, default=0.,
                        help="training caption NMSE relative to image NMSE; 0 keeps legacy route")
    parser.add_argument("--evaluate-caption", action="store_true",
                        help="report held-out caption error even with image-only routing")
    parser.add_argument("--caption-image-slack", type=float,
                        help="instead of weighted selection, improve caption by group swaps "
                             "with at most this relative training image-SSE increase")
    parser.add_argument("--output", type=Path,
                        help="write one immutable diagnostic JSON in addition to stdout")
    args = parser.parse_args()
    import numpy as np

    manifest = json.loads(args.manifest.read_text())
    identity, shape = manifest["export_identity"], manifest["shape"]
    width, image_rows = shape["ane_mlp_end"], identity["region_image_rows"]
    target_width = args.ane_width or width
    if (identity["activation_precision"] != "int8" or
            "regions_image_sq" not in identity["a8_graph"] or
            not 0 <= args.block < 32 or str(args.block) not in manifest["artifacts"] or
            args.group_width < 32 or 10240 % args.group_width or
            width % args.group_width or
            not 0 < target_width <= 10240 - args.group_width or
            target_width % args.group_width or
            args.calibration_rows < 1 or args.heldout_rows < 1 or
            not np.isfinite(args.caption_route_weight) or args.caption_route_weight < 0 or
            (args.caption_image_slack is not None and
             (not np.isfinite(args.caption_image_slack) or args.caption_image_slack < 0 or
              args.caption_route_weight == 0))):
        raise ValueError("requires a fixed-width image-SQ W8A8 manifest and aligned groups")
    calibration, digest = load_calibration(
        args.calibration_dir / f"block{args.block}", shape["buckets"][0], 3840,
        np, pad_rows=args.block < 2)
    if digest != identity["calibration"][str(args.block)]["sha256"]:
        raise ValueError("calibration does not match manifest")
    heldout, heldout_digest = load_calibration(
        args.heldout_dir / f"block{args.block}", shape["buckets"][0], 3840,
        np, pad_rows=args.block < 2)
    train = select_image_rows(calibration, args.calibration_rows, image_rows, np)
    test = select_image_rows(heldout[:2], args.heldout_rows, image_rows, np)
    # The two noise refiner blocks contain no caption rows: their 32 trailing
    # rows are loader padding, not meaningful conditioning activations.
    caption_weight = args.caption_route_weight if args.block >= 2 else 0.
    evaluate_caption = args.block >= 2 and (caption_weight > 0 or args.evaluate_caption)
    caption_train = (np.concatenate([sample[image_rows:].astype(np.float32)
                                     for sample in calibration]) if evaluate_caption else None)
    caption_test = (np.concatenate([sample[image_rows:].astype(np.float32)
                                    for sample in heldout[:2]]) if evaluate_caption else None)

    prefix = (f"noise_refiner.{args.block}" if args.block < 2 else
              f"layers.{args.block - 2}") + ".feed_forward"
    with SafetensorsSource.from_model(args.model) as reader:
        if str(reader.checkpoint) != identity["checkpoint"]:
            raise ValueError("checkpoint differs from manifest")
        gate = reader.tensor(prefix + ".w1.weight", (10240, 3840), np)
        up = reader.tensor(prefix + ".w3.weight", (10240, 3840), np)
        down = reader.tensor(prefix + ".w2.weight", (3840, 10240), np)

    transformed = smooth_partial_ffn(
        gate, up, down, calibration, identity["sq_alpha1"], identity["sq_alpha2"],
        np, identity["sq_hidden_rows"], outlier_rows=True,
        hidden_region_rows=image_rows)
    gate_s, up_s, down_s, s1, _ = transformed
    scale = np.float16(max(float(np.max(np.abs(train / s1))) / 127, 1e-6))
    x = train / s1
    xq = qdq(x, scale, np)
    reference_hidden = swiglu(x, gate_s, up_s, np) / shape["activation_scale"] ** 2
    input_a8_hidden = swiglu(xq, gate_s, up_s, np) / shape["activation_scale"] ** 2
    if not np.isfinite(reference_hidden).all() or not np.isfinite(input_a8_hidden).all():
        raise ValueError("nonfinite calibration hidden state")
    if evaluate_caption:
        caption_x = caption_train / s1
        caption_scale = np.float16(max(float(np.max(np.abs(caption_x))) / 127, 1e-6))
        caption_reference_hidden = swiglu(caption_x, gate_s, up_s, np) / (
            shape["activation_scale"] ** 2)
        caption_a8_hidden = swiglu(qdq(caption_x, caption_scale, np), gate_s, up_s,
                                   np) / (shape["activation_scale"] ** 2)
        if (not np.isfinite(caption_reference_hidden).all() or
                not np.isfinite(caption_a8_hidden).all()):
            raise ValueError("nonfinite caption routing hidden state")
    errors = []
    caption_errors = []
    for start in range(0, 10240, args.group_width):
        end = start + args.group_width
        ref = reference_hidden[:, start:end]
        approximate = input_a8_hidden[:, start:end]
        trial_scale = np.float16(max(float(np.max(np.abs(approximate))) / 127, 1e-6))
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            projected = (qdq(approximate, trial_scale, np) - ref) @ down_s[:, start:end].T
        if not np.isfinite(projected).all():
            raise ValueError("nonfinite channel-group routing error")
        errors.append(projected.astype(np.float32))
        if caption_weight:
            caption_piece = caption_a8_hidden[:, start:end]
            caption_scale = np.float16(max(float(np.max(np.abs(caption_piece))) / 127,
                                           1e-6))
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                projected_caption = (qdq(caption_piece, caption_scale, np) -
                                     caption_reference_hidden[:, start:end]) @ down_s[:, start:end].T
            if not np.isfinite(projected_caption).all():
                raise ValueError("nonfinite caption channel-group routing error")
            caption_errors.append(projected_caption.astype(np.float32))
    image_errors = np.stack(errors)
    swaps = []
    if caption_weight and args.caption_image_slack is not None:
        selected, swaps = constrained_caption_route(
            image_errors, np.stack(caption_errors),
            route_groups(image_errors, target_width // args.group_width, np),
            args.caption_image_slack, np)
    elif caption_weight:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            image_reference = reference_hidden @ down_s.T
            caption_reference = caption_reference_hidden @ down_s.T
        combined = combine_region_errors(
            image_errors, np.stack(caption_errors), image_reference,
            caption_reference, caption_weight, np)
        selected = route_groups(combined, target_width // args.group_width, np)
    else:
        selected = route_groups(image_errors, target_width // args.group_width, np)
    original = list(range(target_width // args.group_width))

    def assess(group_indexes, region="image"):
        indexes = np.concatenate([np.arange(i * args.group_width, (i + 1) * args.group_width)
                                  for i in group_indexes])
        # Re-derive SmoothQuant and A8 thresholds on *training* captures only.
        g, u, d, local_s1, _ = smooth_partial_ffn(
            gate[indexes], up[indexes], down[:, indexes], calibration,
            identity["sq_alpha1"], identity["sq_alpha2"], np,
            identity["sq_hidden_rows"], outlier_rows=True,
            hidden_region_rows=image_rows)
        first, last = ((0, image_rows) if region == "image" else (image_rows, shape["buckets"][0]))
        region_train = [sample[first:last] for sample in calibration]
        region_test = test if region == "image" else caption_test
        input_scale = np.float16(max(float(np.max(np.abs(
            np.concatenate([sample.astype(np.float32) / local_s1
                            for sample in region_train])))) / 127, 1e-6))
        # Hidden A8 scale is trained on calibration rows, never held-out rows.
        hidden_scale, _ = calibrate_hidden_a8(
            g, u, d, local_s1, region_train,
            float(input_scale), shape["activation_scale"], np, outlier_rows=True)
        normalized = region_test / local_s1
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reference = swiglu(normalized, g, u, np) @ d.T
        hidden = swiglu(qdq(normalized, input_scale, np), g, u, np)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            candidate = qdq(hidden / shape["activation_scale"] ** 2,
                            hidden_scale, np) @ d.T * shape["activation_scale"] ** 2
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError("nonfinite routed partial FFN result")
        error = float(np.sum((candidate - reference).astype(np.float64) ** 2))
        return reference, error, float(input_scale), float(hidden_scale)

    original_reference, original_error, _, _ = assess(original)
    selected_reference, selected_error, selected_input, selected_hidden = assess(selected)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        full_reference = swiglu(test, gate.astype(np.float32), up.astype(np.float32), np)
        full_reference = full_reference @ down.astype(np.float32).T
    if not np.isfinite(full_reference).all():
        raise ValueError("nonfinite full FFN reference")
    full_energy = max(float(np.sum(full_reference.astype(np.float64) ** 2)), 1e-12)
    caption_assessment = {}
    if evaluate_caption:
        original_caption, original_caption_error, _, _ = assess(original, "caption")
        selected_caption, selected_caption_error, _, _ = assess(selected, "caption")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            full_caption = swiglu(caption_test, gate.astype(np.float32),
                                  up.astype(np.float32), np) @ down.astype(np.float32).T
        if not np.isfinite(full_caption).all():
            raise ValueError("nonfinite caption full FFN reference")
        caption_energy = max(float(np.sum(full_caption.astype(np.float64) ** 2)), 1e-12)
        caption_assessment = {
            "caption_contiguous_error_vs_full_relative_l2":
                (original_caption_error / caption_energy) ** .5,
            "caption_routed_error_vs_full_relative_l2":
                (selected_caption_error / caption_energy) ** .5,
            "caption_contiguous_partial_relative_l2":
                (original_caption_error / max(float(np.sum(
                    original_caption.astype(np.float64) ** 2)), 1e-12)) ** .5,
            "caption_routed_partial_relative_l2":
                (selected_caption_error / max(float(np.sum(
                    selected_caption.astype(np.float64) ** 2)), 1e-12)) ** .5,
        }
    result = {
        "block": args.block, "group_width": args.group_width,
        "ane_width": target_width, "selected_groups": selected,
        "calibration_source_ane_width": width,
        "caption_route_weight": caption_weight,
        **({"caption_image_slack": args.caption_image_slack,
            "caption_group_swaps": swaps} if args.caption_image_slack is not None else {}),
        "contiguous_partial_relative_l2": (original_error / max(float(np.sum(
            original_reference.astype(np.float64) ** 2)), 1e-12)) ** .5,
        "routed_partial_relative_l2": (selected_error / max(float(np.sum(
            selected_reference.astype(np.float64) ** 2)), 1e-12)) ** .5,
        "contiguous_error_vs_full_relative_l2": (original_error / full_energy) ** .5,
        "routed_error_vs_full_relative_l2": (selected_error / full_energy) ** .5,
        "routed_calibrated_input_a8_scale": selected_input,
        "routed_calibrated_hidden_a8_scale": selected_hidden,
        **caption_assessment,
        "scope": "FP32 A8-only, training-only routing with held-out validation; not W8/CoreML/image",
    }
    if args.output is not None:
        if args.output.is_symlink() or args.output.exists():
            raise ValueError("routing diagnostic output must be a new regular file")
        result["provenance"] = {
            "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "calibration_sha256": digest,
            "heldout_sha256": heldout_digest,
            "training_only_selection": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
