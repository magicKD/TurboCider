#!/usr/bin/env python3
"""Paired production-CLI ConvRot encoder/DiT ablation, with independent memory evidence.

Prepare and validate requests without loading a model:
  .venv/bin/python3 tools/benchmarks/qwen21_convrot_w8a8_benchmark.py \
    --output build/convrot-plan --plan-only

Run 256px smoke requests, then a separate 512px timing experiment:
  ... --output build/convrot-smoke --dimension 256 --smoke
  ... --output build/convrot-perf --dimension 512 --repeats 3

Arms are encoder/DiT bits: 00 reference, 10 encoder only, 01 DiT only, 11 both.
Every arm owns a fresh, sequential CLI process using the same original weights.
Warmup uses a third prompt; measured short/long prompts alternate so conditioning
must be recomputed. --dump-tensors enables a separate correctness experiment;
its timings are deliberately not advertised as performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
sys.path.insert(0, str(ROOT / "tools/validation"))
from quality_gate import image_metrics, passes as image_passes
from runtime_ane_memory import run_sampled, signal_owned_group

CHECKPOINT = "qwen_image_2.1_turbo_int8_convrot.safetensors"
SCHEDULE = "qwen21-turbo-authored-fixed8-cfg1-v1"
SOURCES = (
    "diffusion_models/" + CHECKPOINT,
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
    "processor/tokenizer.json",
)
PROMPTS = {
    "warmup": "A small red apple beside a white porcelain cup, soft window light, still life photograph.",
    "short": "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
    "long": (
        "A realistic editorial photograph of a blue ceramic teapot on an old oak table beside a tall "
        "window in a quiet kitchen. The teapot has a curved handle, a long narrow spout, a fitted lid, "
        "and a subtle handmade glaze with tiny irregular speckles. A folded cream linen napkin, "
        "two empty white porcelain cups, and a small dish of sliced lemons sit nearby. Gentle morning "
        "light enters from the left, making soft shadows and delicate reflections on the glazed "
        "surface. In the background are pale plaster walls, a wooden shelf, and a leafy green plant. "
        "Use natural colors, carefully observed material textures, a balanced composition, a slightly "
        "elevated three-quarter camera angle, and modest depth of field. Keep the complete handle and "
        "spout visible inside the frame, with enough empty space around the arrangement."
    ),
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class BenchmarkMemoryGuard:
    """Stop the owned sampler/CLI group if its measured memory gets too high."""

    def __init__(self, evidence: Path, limit_bytes: int):
        self.evidence = evidence
        self.limit_bytes = limit_bytes
        self.reason = None
        self.done = threading.Event()

    def start(self, pid: int) -> None:
        self.worker = threading.Thread(target=self.watch, args=(pid,), daemon=True)
        self.worker.start()

    def finish(self) -> None:
        self.done.set()
        self.worker.join()

    def watch(self, pid: int) -> None:
        offset = 0
        while not self.done.wait(0.1):
            if not self.evidence.is_file():
                continue
            with self.evidence.open("rb") as samples:
                samples.seek(offset)
                for line in samples:
                    if not line.endswith(b"\n"):
                        break
                    offset += len(line)
                    record = json.loads(line)
                    if record.get("type") != "sample" or not record.get("root_alive"):
                        continue
                    footprint = record["tree_phys_footprint_bytes"]
                    swap_out = record["system"]["swap_out_bytes"]
                    if footprint <= self.limit_bytes and swap_out <= (256 << 20):
                        continue
                    self.reason = {"footprint_bytes": footprint, "limit_bytes": self.limit_bytes,
                                   "system_swap_out_bytes": swap_out}
                    signal_owned_group(pid, signal.SIGTERM)
                    if not self.done.wait(3):
                        signal_owned_group(pid, signal.SIGKILL)
                    return


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot(model: Path, *, required: bool) -> dict:
    result = {}
    for relative in SOURCES:
        path = model / relative
        if not path.is_file():
            if required:
                raise ValueError(f"missing original model component: {path}")
            result[relative] = {"missing": True}
            continue
        before = path.stat()
        with path.open("rb") as handle:
            if path.suffix == ".safetensors":
                prefix = handle.read(8)
                size = int.from_bytes(prefix, "little")
                if len(prefix) != 8 or not 2 <= size <= 16 << 20 or size + 8 > before.st_size:
                    raise ValueError(f"invalid bounded safetensors header: {path}")
                payload = handle.read(size)
                if len(payload) != size:
                    raise ValueError(f"truncated safetensors header: {path}")
            else:
                payload = handle.read(16 << 20)
                if handle.read(1):
                    raise ValueError(f"processor exceeds bounded snapshot size: {path}")
        after = path.stat()
        generation = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if generation(before) != generation(after):
            raise ValueError(f"model component changed while snapshotting: {path}")
        result[relative] = {
            "path": str(path.resolve()), "generation": list(generation(after)),
            "header_or_processor_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result


def arm_environment(arm: str, inherited=None) -> dict:
    source = os.environ if inherited is None else inherited
    env = {key: value for key, value in source.items() if not key.startswith("TURBOCIDER_")}
    env.update({
        "TURBOCIDER_QWEN21_CONVROT_PACKED": "1",
        "TURBOCIDER_QWEN21_CONVROT_W8A8": "0",
        "TURBOCIDER_QWEN21_CONVROT_W8A8_ENCODER": arm[0],
        "TURBOCIDER_QWEN21_CONVROT_W8A8_DIT": arm[1],
        "TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS": "0",
        "TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME": "0",
    })
    return env


def request_specs(args, directory: Path) -> list[dict]:
    prompts = [("warmup", f"{PROMPTS['warmup']} Arrangement number {i + 1}.") for i in range(args.warmup)]
    prompts += [(case, PROMPTS[case]) for _ in range(args.repeats)
                for case in (("short",) if args.smoke else ("short", "long"))]
    result = []
    for index, (case, prompt) in enumerate(prompts):
        stem = f"{index:02d}-{case}"
        request = {
            "model": "qwen-image-2.1", "model_variant": "qwen-image-2.1-turbo",
            "operation": "image.generate", "prompt": prompt,
            "width": args.dimension, "height": args.dimension, "steps": 8, "seed": args.seed,
            "frames": 1, "audio": False, "execution": "gpu", "residency": args.residency,
            "allow_approximation": True, "qwen21_dit_cache": "off",
            "output": str(directory / (stem + ".png")),
        }
        if args.dump_tensors:
            request["dump_tensors"] = str(directory / (stem + "-tensors"))
        path = directory / (stem + "-request.json")
        write_json(path, request)
        result.append({"index": index, "case": case, "warmup": case == "warmup",
                       "path": str(path), "request": request})
    return result


def decode_cli_records(text: str) -> list[dict]:
    """The CLI currently prints pretty JSON; batch is concatenated JSON, not JSONL."""
    decoder = json.JSONDecoder()
    values = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        if not isinstance(value, dict):
            raise ValueError("CLI result must be a JSON object")
        values.append(value)
    return values


def validate_record(record: dict, spec: dict, arm: str) -> None:
    request = spec["request"]
    for key in ("model", "model_variant", "operation", "width", "height", "steps", "seed", "output"):
        if record.get(key) != request[key]:
            raise ValueError(f"request {spec['index']} changed {key}")
    if record.get("checkpoint") != CHECKPOINT or record.get("actual_denoise_steps") != 8:
        raise ValueError("actual original Turbo checkpoint and all eight denoise steps required")
    if record.get("qwen21_sampling_schedule") != SCHEDULE:
        raise ValueError("actual authored Turbo8 CFG1 sampling schedule required")
    if record.get("prompt_cache_hit") is not False:
        raise ValueError("conditioning cache hit invalidates the encoder comparison")
    if request.get("residency") == "streamed":
        if (record.get("plan") or {}).get("residency") != "streamed":
            raise ValueError("the runtime plan did not retain streamed residency")
        streamed = record.get("qwen_convrot_streaming") or {}
        cache_budget=request.get("qwen21_stream_weight_budget_bytes",0)
        if streamed.get("requested_weight_budget_bytes",0)!=cache_budget:
            raise ValueError("runtime receipt changed the requested weight budget")
        for index, (name, layers, passes) in enumerate((("encoder", 36, 1), ("denoiser", 32, 8))):
            stage = streamed.get(name) or {}
            prefix=stage.get("prefix")
            if (type(prefix) is not int or not 0 <= prefix <= layers-2 or
                    (name=="encoder" or not cache_budget) and prefix!=0):
                raise ValueError(f"{name} has an unauthorized resident prefix")
            prefixes=stage.get("prefix_per_pass",[prefix]*passes)
            if (not isinstance(prefixes,list) or len(prefixes)!=passes or
                    any(type(value) is not int or value not in (0,prefix) for value in prefixes) or
                    prefixes!=sorted(prefixes,reverse=True) or stage.get("final_prefix",prefix)!=prefixes[-1]):
                raise ValueError("invalid cache eviction/pass coverage")
            retained=sum(prefixes)
            mode = "int8_tensorops_w8a8" if arm[index] == "1" else "q8_g128_float_a"
            if (stage.get("source_precision") != "original_i8_convrot" or stage.get("matmul_mode") != mode or
                    stage.get("drained") is not True or stage.get("slots") != 2 or
                    stage.get("peak_live_layer_owners") != 1 or stage.get("completed_passes") != passes or
                    stage.get("completed_layers") != layers*passes or
                    stage.get("completed_prefix_layers") != retained or
                    any(stage.get(key) != layers*passes-retained for key in
                        ("fills", "completed_streamed_layers", "reader_fences", "completed_reader_fences"))):
                raise ValueError(f"{name} did not complete the requested ConvRot streaming schedule")
            raw = stage["managed_weight_capacity_bytes"]
            derived = stage["derived_weight_capacity_bytes"]
            budget = stage["managed_weight_budget_bytes"]
            if cache_budget and budget!=cache_budget:
                raise ValueError("runtime did not retain the requested weight budget")
            if (not 0 < raw + derived <= budget or not stage["streamed_source_bytes"] or
                    (derived != 0 if arm[index] == "1" else derived <= 0)):
                raise ValueError(f"{name} managed streaming capacity is invalid")
    execution = record.get("qwen_convrot_w8a8") or {}
    for index, component in enumerate(("encoder", "dit")):
        enabled = arm[index] == "1"
        if execution.get(component + "_enabled") is not enabled:
            raise ValueError(f"actual {component} mode differs from arm {arm}")
        for suffix in ("_projections", "_activation_rows"):
            count = execution.get(component + suffix)
            if type(count) is not int or (count <= 0 if enabled else count != 0):
                raise ValueError(f"actual {component}{suffix} does not prove the requested execution")
    timings = record.get("timings_seconds") or {}
    for key in ("request_wall", "text_encode", "denoise", "vae_decode"):
        value = timings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"finite positive {key} timing required")
    memory = record.get("memory") or {}
    for key in ("mlx_peak_bytes", "mlx_active_bytes"):
        if type(memory.get(key)) is not int or memory[key] < 0:
            raise ValueError(f"actual {key} memory counter required")
    if not Path(request["output"]).is_file():
        raise ValueError("successful native request did not export its PNG")


def summarize_records(rows: list[dict]) -> dict:
    result = {}
    for case in ("short", "long"):
        selected = [row["metrics"] for row in rows if row["case"] == case and not row["warmup"]]
        if not selected:
            continue
        timings = {}
        for key in selected[0]["timings_seconds"]:
            values = [row["timings_seconds"][key] for row in selected]
            timings[key] = {"values": values, "median": statistics.median(values),
                            "minimum": min(values), "maximum": max(values)}
        result[case] = {"count": len(selected), "timings_seconds": timings,
                        "maximum_mlx_peak_bytes": max(row["memory"]["mlx_peak_bytes"] for row in selected),
                        "text_tokens": [row["text_tokens"] for row in selected]}
    return result


def read_tensor(path: Path):
    """CPU-only bounded tensor reader; never imports MLX to compare dumps."""
    import numpy as np
    if path.stat().st_size > 64 << 20:
        raise ValueError("diagnostic tensor exceeds the 64 MiB comparison bound")
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("truncated diagnostic tensor")
    size = struct.unpack("<Q", data[:8])[0]
    if not 2 <= size <= len(data) - 8:
        raise ValueError("invalid diagnostic tensor header")
    tensor = json.loads(data[8:8 + size])["tensor"]
    begin, end = tensor["data_offsets"]
    payload = data[8 + size:]
    if not 0 <= begin < end <= len(payload):
        raise ValueError("invalid diagnostic tensor offsets")
    dtypes = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}
    values = np.frombuffer(payload[begin:end], dtype=dtypes[tensor["dtype"]])
    if tensor["dtype"] == "BF16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    return values.reshape(tensor["shape"]).astype(np.float64)


def tensor_metrics(reference: Path, candidate: Path) -> dict:
    import numpy as np
    first, second = read_tensor(reference), read_tensor(candidate)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        return {"shape_equal": first.shape == second.shape, "finite": False}
    a, b = first.ravel(), second.ravel()
    delta = a - b
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"shape_equal": True, "finite": True, "exact": bool(np.array_equal(a, b)),
            "relative_l2": float(np.linalg.norm(delta)) / max(norm_a, 1e-30),
            "maximum_absolute": float(np.max(np.abs(delta))),
            "cosine": float(np.dot(a, b)) / (norm_a * norm_b) if norm_a and norm_b else float(np.array_equal(a, b))}


def compare_arms(arms: list[dict], args) -> list[dict]:
    baseline = next((arm for arm in arms if arm["arm"] == "00" and arm["status"] == "complete"), None)
    if baseline is None:
        return []
    result = []
    for candidate in arms:
        if candidate["arm"] == "00" or candidate["status"] != "complete":
            continue
        comparisons = []
        for left, right in zip(baseline["runs"], candidate["runs"]):
            if left["warmup"]:
                continue
            if (left["case"], left["request"]["prompt"], left["request"]["seed"]) != (right["case"], right["request"]["prompt"], right["request"]["seed"]):
                raise ValueError("unpaired output requests")
            pixels = image_metrics(Path(left["request"]["output"]), Path(right["request"]["output"]))
            row = {"index": left["index"], "case": left["case"], "pixels": pixels,
                   "pixel_gate_passed": image_passes(pixels, min_correlation=args.min_correlation,
                       min_cosine=args.min_cosine, max_mae_255=args.max_mae_255)}
            if args.dump_tensors:
                row["tensors"] = {name: tensor_metrics(Path(left["request"]["dump_tensors"]) / (name + ".safetensors"),
                    Path(right["request"]["dump_tensors"]) / (name + ".safetensors"))
                    for name in ("qwen21_text", "qwen21_initial", "qwen21_latents")}
                if not all(value["shape_equal"] and value["finite"] for value in row["tensors"].values()):
                    raise ValueError("paired diagnostic tensors differ in shape or contain nonfinite values")
                if not row["tensors"]["qwen21_initial"]["exact"]:
                    raise ValueError("paired requests did not use identical initial noise")
            comparisons.append(row)
        ratios = {}
        for case, control in baseline["summary"].items():
            alternate = candidate["summary"][case]
            ratios[case] = {key: control["timings_seconds"][key]["median"] / alternate["timings_seconds"][key]["median"]
                           for key in ("request_wall", "text_encode", "denoise", "vae_decode")}
        result.append({"baseline": "00", "candidate": candidate["arm"], "paired_outputs": comparisons,
                       "pixel_gate_passed": all(row["pixel_gate_passed"] for row in comparisons),
                       "speedup_by_phase": ratios})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1-Turbo-INT8")
    parser.add_argument("--cli", type=Path, default=ROOT / "build/native/turbocider")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing evidence is never overwritten")
    parser.add_argument("--arms", nargs="+", choices=("00", "10", "01", "11"), default=["00", "10", "01", "11"])
    parser.add_argument("--dimension", type=int, choices=(128, 256, 512), default=512)
    parser.add_argument("--residency", choices=("streamed", "resident"), default="streamed")
    parser.add_argument("--max-footprint-gib", type=float, default=8,
                        help="stop the benchmark's process group above this sampled physical footprint")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="measured short/long pairs per arm")
    parser.add_argument("--smoke", action="store_true", help="one short prompt per arm, no warmup or performance claim")
    parser.add_argument("--plan-only", action="store_true", help="write requests and run CLI plan; never load model weights")
    parser.add_argument("--dump-tensors", action="store_true", help="separate correctness run; timing includes diagnostic I/O")
    parser.add_argument("--timeout-seconds", type=float, default=1800, help="owned process deadline per arm")
    parser.add_argument("--min-correlation", type=float, default=0.99)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--max-mae-255", type=float, default=5.0)
    parser.add_argument("--require-pixel-gate", action="store_true", help="return failure when the paired pixel gate fails")
    args = parser.parse_args(argv)
    if args.smoke:
        args.warmup, args.repeats = 0, 1
    if not 0 <= args.warmup <= 3 or not 1 <= args.repeats <= 10 or not 0 <= args.seed < 2**64:
        parser.error("warmup must be 0...3, repeats 1...10, and seed uint64")
    if not math.isfinite(args.timeout_seconds) or not 1 <= args.timeout_seconds <= 7200:
        parser.error("timeout must be finite and within 1...7200 seconds")
    if not math.isfinite(args.max_footprint_gib) or not 1 <= args.max_footprint_gib <= 24:
        parser.error("footprint threshold must be finite and within 1...24 GiB")
    if (not all(math.isfinite(value) for value in (args.min_correlation, args.min_cosine, args.max_mae_255))
            or not -1 <= args.min_correlation <= 1 or not -1 <= args.min_cosine <= 1 or args.max_mae_255 < 0):
        parser.error("quality thresholds must be finite, correlation/cosine within -1...1, and MAE nonnegative")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms would overwrite matched evidence")
    args.model, args.cli, args.output = args.model.resolve(), args.cli.resolve(strict=True), args.output.resolve()
    if args.require_pixel_gate and ("00" not in args.arms or len(args.arms) < 2):
        parser.error("the pixel gate requires reference arm 00 and at least one candidate")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "tc-qwen21-convrot-w8a8-benchmark-v1", "status": "running",
        "mode": "plan-only" if args.plan_only else "smoke" if args.smoke else "tensor-correctness" if args.dump_tensors else "timing",
        "dimension": args.dimension, "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "residency": args.residency, "max_footprint_bytes": int(args.max_footprint_gib * (1 << 30)),
        "platform": platform.platform(), "cli": str(args.cli), "cli_sha256": sha256(args.cli),
        "model": str(args.model), "model_identity_scope": "original names, file generations and bounded headers; not full payload authentication",
        "script_sha256": sha256(Path(__file__)), "arms": [], "comparisons": [],
        "quality_thresholds": {"minimum_correlation": args.min_correlation, "minimum_cosine": args.min_cosine, "maximum_mae_255": args.max_mae_255},
        "scope": "single seed and generation only; native request timings and independently sampled process memory including cold verification, warmup and exit",
        "performance_evidence": not (args.plan_only or args.smoke or args.dump_tensors),
    }
    library = args.cli.parent / "libturbocider.dylib"
    if library.is_file():
        report["library_sha256"] = sha256(library)
    try:
        identity = source_snapshot(args.model, required=not args.plan_only)
        report["model_snapshot"] = identity
        for arm in args.arms:
            directory = args.output / arm
            directory.mkdir()
            specs = request_specs(args, directory)
            env = arm_environment(arm)
            if args.residency == "streamed":
                # Start each arm with the same cold registration state. Subsequent
                # requests in that arm use the runtime's verified source proofs.
                registration = directory / "source-registration"
                registration.mkdir(mode=0o700)
                env["TURBOCIDER_SOURCE_REGISTRATION_CACHE"] = str(registration)
            trial = {"arm": arm, "status": "running", "environment": {key: value for key, value in env.items() if key.startswith(("TURBOCIDER_", "MLX_"))}, "requests": specs}
            trial["command"] = [str(args.cli), "batch", str(args.model), *(spec["path"] for spec in specs)]
            report["arms"].append(trial)
            for spec in specs:
                planned = subprocess.run([str(args.cli), "plan", spec["path"]], env=env, cwd=ROOT,
                                         capture_output=True, text=True, timeout=60)
                Path(spec["path"].replace("-request.json", "-plan.json")).write_text(planned.stdout)
                if planned.returncode:
                    raise RuntimeError(f"arm {arm} request {spec['index']} plan failed: {planned.stderr.strip()}")
            if args.plan_only:
                trial["status"] = "planned"
                continue
            if source_snapshot(args.model, required=True) != identity:
                raise ValueError("model generation changed between arms")
            command = trial["command"]
            write_json(args.output / "report.json", report)
            print(f"arm {arm}: {len(specs)} complete Turbo8 requests", flush=True)
            started = time.perf_counter()
            guard = BenchmarkMemoryGuard(directory / "process-memory.jsonl", report["max_footprint_bytes"])
            try:
                with (directory / "stdout.json").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
                    trial["memory"] = run_sampled(command, repo=ROOT, output=directory, stem="process", env=env,
                        stdout=stdout, stderr=stderr, timeout=args.timeout_seconds, interval_ms=100, max_gap_ms=500,
                        observer=guard)
            finally:
                if guard.reason is not None:
                    trial["memory_stop"] = guard.reason
                    raise RuntimeError(f"benchmark memory guard stopped arm {arm}")
            trial["harness_elapsed_seconds"] = time.perf_counter() - started
            records = decode_cli_records((directory / "stdout.json").read_text())
            if len(records) != len(specs):
                raise ValueError("CLI did not return every complete warmup/measured request")
            runs = []
            for spec, metrics in zip(specs, records):
                validate_record(metrics, spec, arm)
                write_json(directory / f"{spec['index']:02d}-metrics.json", metrics)
                runs.append({**spec, "metrics": metrics, "png_sha256": sha256(Path(spec["request"]["output"]))})
            trial.update(status="complete", runs=runs, summary=summarize_records(runs))
            if source_snapshot(args.model, required=True) != identity or sha256(args.cli) != report["cli_sha256"]:
                raise ValueError("model or CLI changed during the experiment")
            if library.is_file() and sha256(library) != report.get("library_sha256"):
                raise ValueError("native library changed during the experiment")
            write_json(args.output / "report.json", report)
        if not args.plan_only:
            report["comparisons"] = compare_arms(report["arms"], args)
        report["status"] = "planned" if args.plan_only else "complete"
        if args.require_pixel_gate and not args.plan_only and not all(pair["pixel_gate_passed"] for pair in report["comparisons"]):
            report.update(status="quality-gate-failed")
    except Exception as error:
        report.update(status="failed", error=str(error))
        if report["arms"] and report["arms"][-1]["status"] == "running":
            report["arms"][-1].update(status="failed", error=str(error))
    write_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"), "error": report.get("error")}, indent=2))
    return 0 if report["status"] in ("planned", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
