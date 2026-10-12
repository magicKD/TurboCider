#!/usr/bin/env python3
"""One opt-in, bounded base pair using existing local weights only.

Supervision/receipt only; production inference is the native CLI, not Python.
No exports, intermediate tensor dumps, warmup loops or downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_bounded(process, timeout):
    """A timeout is a failed experiment even when termination races with exit 0."""
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        pass
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass  # Collect the exit status instead of dropping the failure receipt.
        try:
            return process.wait(timeout=grace), True
        except subprocess.TimeoutExpired:
            pass
    return None, True  # Unknown termination stays a failed, stopped pair.


def system_snapshot():
    result = {}
    for key, command in (
        ("swap", ["/usr/sbin/sysctl", "vm.swapusage"]),
        ("vm", ["/usr/bin/vm_stat"]),
        ("thermal", ["/usr/bin/pmset", "-g", "therm"]),
    ):
        call = subprocess.run(command, capture_output=True, text=True, timeout=5)
        result[key] = dict(returncode=call.returncode, stdout=call.stdout, stderr=call.stderr)
    return result


def source_metadata(root):
    result = {}
    for relative in ("diffusion_models/qwen_image_2.1_bf16.safetensors",
                     "text_encoders/qwen3vl_8b_bf16.safetensors",
                     "vae/qwen_image_2.1_vae_bf16.safetensors", "processor/tokenizer.json"):
        path = root / relative
        stat = path.stat()
        result[relative] = dict(path=str(path.resolve()), bytes=stat.st_size, inode=stat.st_ino,
                                device=stat.st_dev, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
    return result


def pair_routes(comparison, output, weight_staging="generic", weight_scale_cache=False):
    if weight_scale_cache and comparison != "hybrid-vs-native":
        raise ValueError("weight scale cache requires the hybrid-vs-native comparison")
    """Keep the old GPU/MLX oracle pair; W1 is a separate explicit experiment."""
    if comparison == "hybrid-vs-native":
        return [
            dict(name="native", backend="native-metal-graph", execution="gpu",
                 allow_approximation=True),
            dict(name="hybrid", backend="native-metal-graph", execution="gpu_ane",
                 allow_approximation=True, qwen21_w8a8=True,
                 qwen21_native_ane_cache=str(output / "owned-ane-cache"),
                 qwen21_native_w8_staging=weight_staging,
                 qwen21_native_scale_cache=weight_scale_cache),
        ]
    return [
        dict(name="native", backend="native-metal-graph", execution="gpu", allow_approximation=True),
        dict(name="mlx", backend="mlx", execution="gpu", allow_approximation=False),
    ]


def verify_native_result(result, route):
    if route["backend"] != "native-metal-graph":
        return
    if result.get("actual_denoise_steps") != 20:
        raise ValueError("native result did not complete all 20 denoise steps")
    phase = result.get("native_denoiser_memory_phase", {}).get("memory_admission", {})
    if phase.get("execution_state") != "succeeded" or phase.get("worker_quarantined") or phase.get("tainted"):
        raise ValueError("native phase did not finish successfully")
    for key in ("ledger_storage_count", "ledger_storage_bytes", "ledger_reserved_bytes",
                "ledger_pending_release_bytes", "ledger_unknown_bytes"):
        if phase.get(key) != 0:
            raise ValueError(f"native phase retained or failed to report {key}")
    if route["execution"] == "gpu_ane":
        metrics = result.get("native_denoiser_hybrid", {})
        if (metrics.get("enabled") is not True or metrics.get("successful_ffns") != 608
                or metrics.get("completed_hybrid_steps") != 19):
            raise ValueError("native W1 did not finish exactly 608 FFNs / 19 hybrid steps")
        if result.get("runtime_backend") != "native_metal_graph+private_ane_w1+mlx_encoder_vae":
            raise ValueError("executed runtime backend is not native private W1")
        if metrics.get("weight_staging") != route.get("qwen21_native_w8_staging", "generic"):
            raise ValueError("executed W8 staging differs from the explicit experiment")
        cache = metrics.get("weight_scale_cache", {})
        expected_cache = route.get("qwen21_native_scale_cache", False)
        if expected_cache:
            expected = dict(entries=32, payload_bytes=1048576, hits=576, misses=32)
            if cache.get("enabled") is not True or any(type(cache.get(key)) is not int or
                    cache[key] != value for key, value in expected.items()):
                raise ValueError("native scale cache did not report exactly 32 fills / 576 hits")
        elif cache and (cache.get("enabled") is not False or any(cache.get(key) != 0
                         for key in ("entries", "payload_bytes", "hits", "misses"))):
            raise ValueError("native scale cache executed despite being disabled")
    elif result.get("runtime_backend") != "native_metal_graph+mlx_encoder_vae":
        raise ValueError("executed runtime backend is not native GPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--comparison", choices=("native-vs-mlx", "hybrid-vs-native"),
                        default="native-vs-mlx")
    parser.add_argument("--weight-staging", choices=("generic", "register"), default="generic")
    parser.add_argument("--weight-scale-cache", action="store_true",
                        help="explicit request-local cache candidate; no extra model files")
    args = parser.parse_args()
    if not 30 <= args.timeout_seconds <= 150:
        parser.error("timeout must be 30..150 seconds per single request")
    if args.weight_staging != "generic" and args.comparison != "hybrid-vs-native":
        parser.error("register staging requires the hybrid-vs-native comparison")
    if args.weight_scale_cache and args.comparison != "hybrid-vs-native":
        parser.error("weight scale cache requires the hybrid-vs-native comparison")
    native, model, output = args.native_dir.resolve(), args.model_root.resolve(), args.output_dir.resolve()
    # A fresh result directory protects prior evidence and keeps this one pair
    # distinct. Models and system/driver caches are never touched.
    output.mkdir(parents=True, exist_ok=False)
    cli, library = native / "turbocider", native / "libturbocider.dylib"
    build = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(("TURBOCIDER_QWEN21_", "TURBOCIDER_ANE_CALIBRATION_")):
            env.pop(key)
    routes = pair_routes(args.comparison, output, args.weight_staging, args.weight_scale_cache)
    receipt = dict(schema="qwen-native-base-e2e-pair-v1", full_request_certified=False,
                   comparison=args.comparison,
                   order=[r["name"] for r in routes], routes=routes, samples_per_route=1,
                   cache_state="fresh process; shared OS source/driver caches not reset",
                   runtime_build_id=build["runtime_build_id"], cli_sha256=digest(cli),
                   library_sha256=digest(library), source_before=source_metadata(model),
                   before=system_snapshot(), runs=[])
    receipt_path = output / "receipt.json"
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    for route in routes:
        name, backend = route["name"], route["backend"]
        request = dict(model="qwen-image-2.1", operation="image.generate",
                       prompt="A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
                       output=str(output / f"{name}.png"), width=512, height=512, steps=20, seed=42,
                       frames=1, audio=False, execution=route["execution"], residency="component_staged",
                       allow_approximation=route["allow_approximation"], qwen21_dit_cache="off",
                       qwen21_denoiser_backend=backend)
        for key in ("qwen21_w8a8", "qwen21_native_ane_cache", "qwen21_native_w8_staging",
                    "qwen21_native_scale_cache"):
            if key in route:
                request[key] = route[key]
        request_path = output / f"{name}-request.json"
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        command = ["/usr/bin/time", "-l", str(cli), "generate", str(model), str(request_path)]
        run = dict(name=name, backend=backend, request=request, command=command, before=system_snapshot())
        print(f"start={name} timeout={args.timeout_seconds}s", flush=True)
        started = time.monotonic()
        with (output / f"{name}-stdout.json").open("w") as stdout, (output / f"{name}-stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            process_code, timed_out = wait_bounded(process, args.timeout_seconds)
        code = 1 if timed_out else process_code
        run.update(process_returncode=process_code, returncode=code, timeout=timed_out,
                   termination_observed=process_code is not None,
                   supervised_wall_seconds=time.monotonic() - started,
                   after=system_snapshot())
        if code == 0:
            try:
                run["result"] = json.loads((output / f"{name}-stdout.json").read_text())
                run["image_sha256"] = digest(output / f"{name}.png")
                verify_native_result(run["result"], route)
                run["executed_route_and_released_phase_verified"] = backend == "native-metal-graph"
            except (ValueError, FileNotFoundError) as error:
                run["result_error"] = str(error)
                code = 1
                run["returncode"] = code
        receipt["runs"].append(run)
        receipt["source_after"] = source_metadata(model)
        receipt["sources_unchanged"] = receipt["source_before"] == receipt["source_after"]
        save()
        print(f"done={name} returncode={code} wall={run['supervised_wall_seconds']:.3f}s", flush=True)
        if code != 0 or not receipt["sources_unchanged"]:
            print("pair_stopped=1; no further model requests", flush=True)
            return 1
    receipt["after"] = system_snapshot()
    receipt["pair_complete"] = True
    save()
    print(f"pair_complete=1 receipt={receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
