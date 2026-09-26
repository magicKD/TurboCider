#!/usr/bin/env python3
"""Replay captured production Stage-2 QKV against dense and sparse Metal cores."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def replay_keep_budgets(rows):
    """Include matched block fractions alongside historical absolute budgets.

    Safety regions can increase actual retained work; the probe reports that
    fraction separately. These budgets are not attention probability masses.
    """
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 16384:
        raise ValueError("rows must be an integer in 1...16384")
    blocks = (rows + 63) // 64
    return sorted({16, 32, 64, 128} |
                  {max(1, math.ceil(blocks * fraction)) for fraction in (.25, .5, .75)})


def validate_head_metrics(row):
    """Reject stale probes and inconsistent head-wise error accounting."""
    heads = row.get("per_head", [])
    if len(heads) != row["heads"]:
        raise ValueError("missing head metrics; rebuild the attention probe")
    error = norm = 0.0
    for index, head in enumerate(heads):
        if head["head"] != index:
            raise ValueError("head metrics are missing, duplicated or out of order")
        for name in ("squared_error", "reference_squared_norm", "max_abs", "exact_block_fraction"):
            if not math.isfinite(head[name]) or head[name] < 0:
                raise ValueError(f"invalid per-head {name}")
        if head["exact_block_fraction"] > 1:
            raise ValueError("invalid per-head retained fraction")
        expected = math.sqrt(head["squared_error"] / head["reference_squared_norm"]) if head["reference_squared_norm"] else None
        actual = head["relative_l2"]
        if ((expected is None and actual is not None) or
                (expected is not None and (actual is None or not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-10)))):
            raise ValueError("inconsistent per-head relative error")
        error += head["squared_error"]
        norm += head["reference_squared_norm"]
    if not math.isclose(math.sqrt(error / max(norm, 1e-30)), row["relative_l2"], rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError("head errors do not reconstruct global relative error")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report exists; preserve previous measurements")
    if not 1 <= args.runs <= 100:
        parser.error("runs must be 1...100")
    fixture = args.fixture.resolve()
    metadata = json.loads((fixture / "metadata.json").read_text())
    if (metadata["schema"] != "ltx-qkv-replay-v1" or metadata["dtype"] != "bf16" or
            metadata["qkv_layout"] != "row-head-dim" or metadata["dim"] != 128 or
            metadata["rope_layout"] != "head-row-halfdim"):
        parser.error("unsupported capture schema/layout")
    root = Path(__file__).resolve().parents[2]
    binary = root / "build/native/ltx-attention-probe"
    shader = root / "native/models/ltx_runtime/ltx_shaders.metal"
    names = ("query.bf16", "key.bf16", "value.bf16", "cosine.bf16", "sine.bf16", "gate.bf16")
    report = dict(schema="ltx-qkv-replay-matrix-v1", fixture=str(fixture), metadata=metadata,
                  fixture_sha256={name: digest(fixture/name) for name in names},
                  binary_sha256=digest(binary), shader_sha256=digest(shader),
                  scope="single captured step/block attention core; not final video quality",
                  rows=[], complete=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    save()
    configs = [("all-exact", 1, .5, 256, 0, 0, 0)]
    configs += [("sol", 0, tau, 1, 0, 0, 0) for tau in (-.5, 0., .5, 1.)]
    configs += [("temporal-window", mode, .5, radius, 16, metadata["tokens_per_frame"], 0)
                for mode in (1, 2, 3) for radius in (1, 2, 4)]
    keep_budgets = replay_keep_budgets(metadata["rows"])
    configs += [("topk", 4, .5, 0, 0, 0, keep) for keep in keep_budgets]
    configs += [("pooled-topk", 5, .5, 0, 0, 0, keep) for keep in keep_budgets]
    for label, mode, tau, radius, anchors, frame_tokens, keep in configs:
        command = [str(binary), str(shader), str(metadata["rows"]), str(metadata["heads"]),
                   str(args.runs), str(tau), str(mode), str(radius), str(anchors),
                   str(frame_tokens), str(keep), str(fixture)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        row = json.loads(result.stdout)
        validate_head_metrics(row)
        row.update(label=label, command=command)
        report["rows"].append(row)
        save()
        print(json.dumps(row), flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
