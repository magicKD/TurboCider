"""Matched single-block Core ML ABBA probe for alternative A8 FFN graphs.

The compute plan is a *preference*, not proof of physical ANE execution.
The source packages are compiled into a fresh temporary directory. Compile,
load and first-prediction costs are excluded from steady-state timings.
"""

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time


def artifact(manifest_path, block):
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["shape"]["buckets"]
    if len(rows) != 1 or rows[0] not in (1024, 1056) or manifest["shape"]["K"] != 3840:
        raise ValueError("expected fixed 1024/1056-row Z-Image artifact")
    return manifest_path.parent / manifest["artifacts"][str(block)]["int8_pc"], rows[0]


def preferred_devices(plan):
    result = {}
    for function in plan.model_structure.program.functions.values():
        for op in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            device = type(usage.preferred_compute_device).__name__
            result.setdefault(op.operator_name, []).append(device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="source manifest")
    parser.add_argument("--candidate", type=Path, required=True, help="source manifest")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 2 or args.warmups < 1 or not 0 <= args.block < 32:
        parser.error("expected positive warmup, at least two rounds, valid block")

    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    import numpy as np

    capture = np.load(args.capture, allow_pickle=False)
    if capture.shape != (1056, 3840) or capture.dtype != np.float16:
        raise ValueError("expected one held-out [1056, 3840] FP16 FFN input")
    sources = {"baseline": artifact(args.baseline, args.block),
               "candidate": artifact(args.candidate, args.block)}
    inputs = {key: np.ascontiguousarray(capture[:rows].T[None, :, None, :])
              for key, (_, rows) in sources.items()}
    with tempfile.TemporaryDirectory(prefix="z-image-a8-abba-") as temporary:
        root = Path(temporary)
        paths = {key: Path(ct.models.utils.compile_model(str(path),
                   destination_path=str(root / f"{key}.mlmodelc")))
                 for key, (path, _) in sources.items()}
        units = ct.ComputeUnit.CPU_AND_NE
        plans = {key: preferred_devices(MLComputePlan.load_from_path(
                    str(path), compute_units=units)) for key, path in paths.items()}
        models = {key: ct.models.CompiledMLModel(str(path), compute_units=units)
                  for key, path in paths.items()}
        for _ in range(args.warmups):
            for key, model in models.items():
                model.predict({"x": inputs[key]})
        samples = {key: [] for key in models}
        outputs = {}
        for _ in range(args.rounds):
            for key in ("baseline", "candidate", "candidate", "baseline"):
                start = time.perf_counter()
                outputs[key] = models[key].predict({"x": inputs[key]})["y"]
                samples[key].append((time.perf_counter() - start) * 1000)
        baseline_image = outputs["baseline"][..., :1024].astype(np.float32)
        candidate_image = outputs["candidate"][..., :1024].astype(np.float32)
        delta = candidate_image - baseline_image
        print(json.dumps({
            "scope": "single-block warmed CPU_AND_NE ABBA; preferred device is not runtime trace",
            "block": args.block, "rounds": args.rounds,
            "rows": {key: rows for key, (_, rows) in sources.items()},
            "medians_milliseconds": {key: statistics.median(v) for key, v in samples.items()},
            "minima_milliseconds": {key: min(v) for key, v in samples.items()},
            "image_output_relative_l2_between_candidates": float(
                np.linalg.norm(delta.ravel()) /
                max(np.linalg.norm(baseline_image.ravel()), 1e-12)),
            "planned_devices": plans,
        }))


if __name__ == "__main__":
    main()
