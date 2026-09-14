"""Non-overwriting, machine-tagged public Core ML Transformer reproduction.

Run serially with other hardware benchmarks stopped. Keeps models and raw
output for audit; does not modify the historical M4 Pro measurements.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return dict(command=command, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def validate_records(rows, iterations):
    """Reject absent/invalid evidence; completion is not a quality pass."""
    samples = [row for row in rows if row.get("record_type") == "samples"]
    summaries = [row for row in rows if row.get("benchmark") == "prefill_transformer_block"]
    if len(samples) != 1 or len(summaries) != 1:
        raise ValueError("expected exactly one sample and summary record")
    for key in ("gpu_ms", "hetero_ms"):
        values = samples[0].get(key, [])
        if len(values) != iterations or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"invalid paired samples: {key}")
    summary = summaries[0]
    for key in ("nrmse", "cosine", "gpu_reference_nrmse", "gpu_reference_cosine",
                "hetero_reference_nrmse", "hetero_reference_cosine",
                "gpu_wall_p50_ms", "hetero_wall_p50_ms", "speedup",
                "up_ane_output_backing_rate"):
        value = summary.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"invalid summary metric: {key}")
    if summary.get("reference_enabled") is not True:
        raise ValueError("independent reference missing")
    if min(summary["gpu_wall_p50_ms"], summary["hetero_wall_p50_ms"], summary["speedup"]) <= 0:
        raise ValueError("nonpositive duration/speed ratio")
    ratio = summary["gpu_wall_p50_ms"] / summary["hetero_wall_p50_ms"]
    if not math.isclose(ratio, summary["speedup"], rel_tol=1e-4):
        raise ValueError("speed ratio inconsistent with durations")
    return dict(
        finite_metrics=True,
        reference_gate_passed=all(summary[f"{device}_reference_nrmse"] <= .01
                                  and summary[f"{device}_reference_cosine"] >= .999
                                  for device in ("gpu", "hetero")),
        output_backing_all_used=summary["up_ane_output_backing_rate"] == 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", default="64x256,256x1024,1024x1024")
    parser.add_argument("--shares", default="0.25,0.5,0.75")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--backend", choices=("public", "private-gpu"), default="public")
    parser.add_argument("--qkv-split", action="store_true",
                        help="split semantic Q to GPU and K/V to ANE")
    args = parser.parse_args()
    if args.qkv_split and args.backend != "public":
        parser.error("private wrapper currently supports FFN only")
    shapes = [tuple(map(int, item.split("x"))) for item in args.shapes.split(",")]
    shares = list(map(float, args.shares.split(",")))
    if (any(len(shape) != 2 or min(shape) <= 0 or shape[1] % 64 for shape in shapes)
            or any(not 0 < share < 1 for share in shares)
            or min(args.layers, args.rounds, args.warmup, args.iterations) < 1):
        parser.error("positive shapes/counts, H divisible by 64, shares in (0,1) required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / ("build/transformer-private-gpu" if args.backend == "private-gpu"
                     else "build/transformer")
    source = ROOT / ("src/runner_private_gpu.mm" if args.backend == "private-gpu"
                     else "src/runner.mm")
    # Record even a failed setup, without inventing hardware metadata.
    metadata = dict(schema="transformer-machine-campaign-v1", phase="initialized",
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    settings={k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()},
                    hardware=capture(["system_profiler", "SPHardwareDataType",
                                      "SPDisplaysDataType", "-json"]),
                    os=capture(["sw_vers"]),
                    power=capture(["pmset", "-g", "batt"]),
                    compiler=capture(["xcrun", "clang++", "--version"]),
                    versions={name: importlib.metadata.version(name)
                              for name in ("numpy", "coremltools")},
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [source, Path(__file__), ROOT / "src/private_ffn.inc",
                                       ROOT / "scripts/private_mil.py",
                                       ROOT / "scripts/add_private_gpu.py",
                                       ROOT / "scripts/prepare.py", ROOT / "scripts/reference.py"]})
    # Hardware IDs are unnecessary to reproduce performance; do not store them.
    try:
        hardware = json.loads(metadata["hardware"]["stdout"])
        for device in hardware.get("SPHardwareDataType", []):
            for key in list(device):
                if any(word in key.lower() for word in ("serial", "uuid", "udid")):
                    del device[key]
        metadata["hardware"]["stdout"] = json.dumps(hardware)
    except json.JSONDecodeError:
        metadata["hardware"]["stdout"] = "unavailable"
    write_json(output / "manifest.json", metadata)
    from prepare import prepare
    from reference import reference
    results = []
    for m, h in shapes:
        for share in shares:
            f = 4 * h
            af = int(f * share) // 64 * 64
            if not 0 < af < f:
                raise ValueError("share rounded to empty branch")
            job = output / f"m{m}_h{h}_af{af}_L{args.layers}"
            job.mkdir()
            cache = job / "models"
            prepare(cache, m, h, f, af, 0, args.layers, "supergraph", m // 2,
                    qkv_split=args.qkv_split)
            if args.backend == "private-gpu":
                from private_mil import private_mil
                private_mil(cache, m, h, f, af, args.layers)
            reference(cache, m, h, f, args.layers, h // 64)
            for round_index in range(args.rounds):
                command = [str(binary), "--stack-root", str(cache),
                           "--layers", str(args.layers), "--m", str(m), "--h", str(h),
                           "--f", str(f), "--heads", str(h // 64), "--kv-heads", str(h // 64),
                           "--qkv-gpu-n", str(h if args.qkv_split else 3*h), "--qkv-cpu-n", "0",
                           "--qkv-ane-n", str(2*h if args.qkv_split else 0),
                           "--up-gpu-n", str(2*(f-af)), "--up-cpu-n", "0", "--up-ane-n", str(2*af),
                           "--ffn-layout", "supergraph", "--coreml-output", "backing",
                           "--attention", "sdpa", "--mode", "both",
                           "--reference-output", str(cache / f"reference_L{args.layers}.fp16"),
                           "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
                if args.qkv_split:
                    command += ["--qkv-model", str(cache / "layer_00/qkv.mlmodelc"),
                                "--qkv-weights", str(cache / "layer_00/qkv_ane.weights.fp16")]
                start = time.perf_counter()
                env = os.environ.copy()
                # No implicit private/serialized ablation inherited from a shell.
                env = {k: v for k, v in env.items() if not k.startswith("TC_")}
                if args.backend == "private-gpu":
                    env["TC_PRIVATE_FFN"] = "1"
                    temporary = job / f"private-tmp-{round_index}"
                    temporary.mkdir()
                    env["TMPDIR"] = str(temporary) + "/"
                run = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     capture_output=True, timeout=600)
                record = dict(command=command, returncode=run.returncode,
                              elapsed_seconds=time.perf_counter()-start,
                              stdout=run.stdout, stderr=run.stderr)
                write_json(job / f"round-{round_index}.json", record)
                if run.returncode:
                    raise RuntimeError(f"benchmark failed; inspect {job}")
                rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
                validation = validate_records(rows, args.iterations)
                result = dict(shape=[m,h,f], ane_width=af, layers=args.layers,
                              round=round_index, records=rows, validation=validation)
                results.append(result)
                print(json.dumps(result), flush=True)
    write_json(output / "results.json", dict(complete=True, results=results,
               scope="Synthetic full causal block; not LTX, not proof of ANE placement"))


if __name__ == "__main__":
    main()
