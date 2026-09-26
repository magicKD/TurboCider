"""Select per-block hidden SmoothQuant alpha using two training captures.

Split the capture directory by recording process ID, fit SQ and the adaptive
A8 scale bank on one capture, and score the other. Reverse the folds before
selecting an alpha. Never use an independent held-out prompt to choose it.
The resulting ranking is an FP32 A8-only screen, not a Core ML or image test.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coreml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_z_image import SafetensorsSource
from z_image_smoothquant import (load_calibration, smooth_partial_ffn,
                                 calibrate_adaptive_hidden_a8, representative_rows)
from z_image_w8a8_layers import measure_a8


def capture_folds(directory, rows, hidden, np, pad_rows=False):
    paths = sorted(directory.glob("*.npy"))
    samples, digest = load_calibration(directory, rows, hidden, np, pad_rows=pad_rows)
    if len(paths) != len(samples):
        raise ValueError("cross-validation requires one array per capture file")
    groups = {}
    for path, sample in zip(paths, samples):
        match = re.fullmatch(r"sample-(\d+)-\d+-\d+\.npy", path.name)
        if match is None:
            raise ValueError("cross-validation requires PID-tagged capture filenames")
        groups.setdefault(match.group(1), []).append(sample)
    if len(groups) != 2 or min(map(len, groups.values())) < 2:
        raise ValueError("cross-validation needs two independent capture processes")
    return list(groups.values()), list(groups), digest


def score_alpha(gate, up, down, train, valid, alpha1, alpha2, activation_scale,
                image_rows, bins, sq_hidden_rows, rows_per_sample, np):
    g, u, d, s1, _ = smooth_partial_ffn(
        gate, up, down, train, alpha1, alpha2, np, sq_hidden_rows,
        outlier_rows=True, hidden_region_rows=image_rows)
    image_train = [sample[:image_rows] for sample in train]
    input_scale = float(np.float16(max(max(
        float(np.max(np.abs(sample.astype(np.float32) / s1)))
        for sample in image_train) / 127, 1e-6)))
    bank = calibrate_adaptive_hidden_a8(
        g, u, s1, image_train, input_scale, activation_scale,
        image_rows, bins, np)
    error = energy = 0.
    for sample in valid:
        image = sample[:image_rows]
        selected = representative_rows(image, rows_per_sample, np)
        result = measure_a8(image[selected], g, u, d, s1, input_scale,
                            float(bank[-1]), activation_scale, np,
                            adaptive_scales=bank, adaptive_shared=True)
        error += result["adaptive_shared_a8_squared_error"]
        energy += result["reference_energy"]
    return error, energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--heldout-dir", type=Path,
                        help="independent capture to report final selected-vs-original scores")
    parser.add_argument("--heldout-samples", type=int, default=4)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--alphas", default="0,0.25,0.4,0.5,0.6,0.75,1")
    parser.add_argument("--rows-per-sample", type=int, default=32)
    args = parser.parse_args()
    import numpy as np

    manifest = json.loads(args.manifest.read_text())
    identity = manifest["export_identity"]
    shape = manifest["shape"]
    if (identity.get("activation_precision") != "int8" or
            not identity.get("adaptive_hidden_a8_shared_qdq") or
            identity.get("calibration_directory_count", 1) != 1 or
            identity.get("checkpoint_format") != "safetensors" or
            len(shape["buckets"]) != 1 or not args.calibration_dir.is_dir() or
            args.calibration_dir.is_symlink() or args.rows_per_sample < 1 or
            args.heldout_samples < 1 or
            (args.heldout_dir is not None and
             (args.heldout_dir.is_symlink() or not args.heldout_dir.is_dir() or
              args.heldout_dir.resolve() == args.calibration_dir.resolve()))):
        raise ValueError("requires a single-directory adaptive W8A8 training manifest")
    alphas = [float(part) for part in args.alphas.split(",")]
    if not alphas or len(set(alphas)) != len(alphas) or any(
            not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError("candidate alphas must be unique and in [0, 1]")
    blocks = sorted(map(int, manifest["artifacts"])) if args.blocks == "all" else [
        int(part) for part in args.blocks.split(",")]
    if len(set(blocks)) != len(blocks) or any(
            str(block) not in manifest["artifacts"] for block in blocks):
        raise ValueError("invalid exported block indexes")
    width, hidden, rows = shape["ane_mlp_end"], shape["K"], shape["buckets"][0]
    image_rows = identity["region_image_rows"]
    bins = identity["adaptive_hidden_a8_bins"]
    with SafetensorsSource.from_model(args.model) as reader:
        if str(reader.checkpoint) != identity["checkpoint"]:
            raise ValueError("model checkpoint does not match export")
        for block in blocks:
            folds, process_ids, digest = capture_folds(
                args.calibration_dir / f"block{block}", rows, hidden, np,
                pad_rows=block < 2)
            if digest != identity["calibration"][str(block)]["sha256"]:
                raise ValueError(f"block {block} training capture differs from export")
            prefix = (f"noise_refiner.{block}" if block < 2 else
                      f"layers.{block - 2}") + ".feed_forward"
            gate = reader.tensor(prefix + ".w1.weight", (10240, hidden), np)[:width]
            up = reader.tensor(prefix + ".w3.weight", (10240, hidden), np)[:width]
            down = reader.tensor(prefix + ".w2.weight", (hidden, 10240), np)[:, :width]
            scores = {}
            for alpha in alphas:
                error = energy = 0.
                for train, valid in ((folds[0], folds[1]), (folds[1], folds[0])):
                    fold_error, fold_energy = score_alpha(
                        gate, up, down, train, valid, identity["sq_alpha1"], alpha,
                        shape["activation_scale"], image_rows, bins,
                        identity["sq_hidden_rows"], args.rows_per_sample, np)
                    error += fold_error
                    energy += fold_energy
                scores[str(alpha)] = (error / max(energy, 1e-12)) ** .5
            selected = min(alphas, key=lambda alpha: (scores[str(alpha)],
                                                       abs(alpha - identity["sq_alpha2"])))
            heldout_result = {}
            if args.heldout_dir is not None:
                heldout, heldout_digest = load_calibration(
                    args.heldout_dir / f"block{block}", rows, hidden, np,
                    pad_rows=block < 2)
                heldout = heldout[:args.heldout_samples]
                for label, alpha in (("selected", selected),
                                     ("original", identity["sq_alpha2"])):
                    error, energy = score_alpha(
                        gate, up, down, folds[0] + folds[1], heldout,
                        identity["sq_alpha1"], alpha, shape["activation_scale"],
                        image_rows, bins, identity["sq_hidden_rows"],
                        args.rows_per_sample, np)
                    heldout_result[f"heldout_{label}_relative_l2"] = (
                        error / max(energy, 1e-12)) ** .5
                heldout_result["heldout_sha256"] = heldout_digest
            print(json.dumps({"block": block, "process_ids": process_ids,
                              "calibration_sha256": digest,
                              "crossval_alpha2": selected,
                              "crossval_relative_l2": scores,
                              **heldout_result,
                              "scope": "two-fold training-only selection; independent "
                                       "FP32 A8-only holdout, not image quality"}), flush=True)


if __name__ == "__main__":
    main()
