#!/usr/bin/env python3
"""Opt-in single MLX/native GPU LoRA pair; inference remains in the native CLI.

Uses the existing pinned Viggle r128 adapter and local model. No downloads,
exports, tensor dumps, warmups, model copies or automatic retries.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import time
import zlib


ADAPTER_NAME = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
ADAPTER_SHA256 = "bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b"
PROMPT = "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph."
MODEL_FILES = (
    "diffusion_models/qwen_image_2.1_bf16.safetensors",
    "text_encoders/qwen3vl_8b_bf16.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors", "processor/tokenizer.json",
)
ROUTES = (
    {"name": "mlx", "backend": "mlx", "runtime_backend": "mlx_cpp_metal"},
    {"name": "native", "backend": "native-metal-graph",
     "runtime_backend": "native_metal_graph+mlx_encoder_vae"},
)
ZERO_LEDGER = ("ledger_storage_count", "ledger_storage_bytes", "ledger_reserved_bytes",
               "ledger_pending_release_bytes", "ledger_unknown_bytes")
LORA_RECIPE = "runtime-lora-tiled8x32-fp32-fma-low-fp32-scaled-ordered-bf16-bias-last-v1"
LORA_SGMMA_RECIPE = "runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2"
LORA_RECIPES = {"scalar": LORA_RECIPE, "sgmma": LORA_SGMMA_RECIPE}
STUDENT_APPROXIMATION = "qwen21_viggle_v021_r128_6step_distillation"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    """Only call for small build/output artifacts, never model/adapter payloads."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_json(path, limit=64 * 1024 * 1024):
    require(path.stat().st_size <= limit, "JSON artifact exceeds bounded receipt size")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("non-finite JSON number: " + value)
    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "JSON result must be an object")
    return value


def source_metadata(model, adapter):
    """Stat the four original assets plus adapter; runtime verifies the pin."""
    result = {}
    for logical, path in [*((relative, model / relative) for relative in MODEL_FILES),
                          ("adapter", adapter)]:
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size > 0,
                "source must be an existing nonempty regular file: " + str(path))
        result[logical] = dict(path=str(path.absolute()), canonical_path=str(path.resolve()),
                               bytes=info.st_size, inode=info.st_ino, device=info.st_dev,
                               mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    return result


def build_metadata(native, expected):
    require(re.fullmatch(r"tc-runtime-build-v1-[0-9a-f]{64}", expected),
            "expected runtime build identity must be explicit and complete")
    manifest = native / "runtime-build/runtime-build-manifest.json"
    build = load_json(manifest, 2 * 1024 * 1024)
    require(build.get("runtime_build_id") == expected, "runtime manifest does not match the pinned build")
    cli, library = native / "turbocider", native / "libturbocider.dylib"
    require(cli.is_file() and os.access(cli, os.X_OK) and library.is_file(),
            "the same build directory must contain an executable CLI and library")
    return dict(runtime_build_id=expected, manifest_sha256=digest(manifest),
                cli_sha256=digest(cli), library_sha256=digest(library))


def clean_environment():
    environment = os.environ.copy()
    removed = []
    for key in tuple(environment):
        if key.startswith(("TURBOCIDER_", "DYLD_")):
            removed.append(key)
            environment.pop(key)
    return environment, sorted(removed)


def system_snapshot():
    result = {}
    for key, command in (("swap", ["/usr/sbin/sysctl", "vm.swapusage"]),
                         ("vm", ["/usr/bin/vm_stat"]),
                         ("thermal", ["/usr/bin/pmset", "-g", "therm"])):
        try:
            call = subprocess.run(command, capture_output=True, text=True, timeout=5)
            result[key] = dict(returncode=call.returncode, stdout=call.stdout, stderr=call.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            result[key] = dict(unavailable=True, error=str(error))
    return result


def wait_bounded(process, timeout):
    """Signals only the owned start_new_session group; never a system helper.

    Observed process termination, including exit 0 after timeout, does not
    certify device completion. A failed/unknown termination stops the pair.
    """
    require(type(timeout) is int and 30 <= timeout <= 120, "timeout must be 30..120 seconds")
    timed_out = cancelled = False
    signals = []
    try:
        return dict(process_returncode=process.wait(timeout=timeout), timeout=False,
                    cancelled=False, termination_observed=True, signals=signals)
    except subprocess.TimeoutExpired:
        timed_out = True
    except KeyboardInterrupt:
        cancelled = True
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
            signals.append(sig.name)
        except ProcessLookupError:
            signals.append(sig.name + ":already-exited")
        try:
            code = process.wait(timeout=grace)
            return dict(process_returncode=code, timeout=timed_out, cancelled=cancelled,
                        termination_observed=True, signals=signals)
        except subprocess.TimeoutExpired:
            continue
    return dict(process_returncode=None, timeout=timed_out, cancelled=cancelled,
                termination_observed=False, signals=signals)


def request_for(route, adapter, output, native_lora_kernel="scalar"):
    require(route in ROUTES, "only the fixed MLX/native GPU routes are permitted")
    require(native_lora_kernel in LORA_RECIPES, "native LoRA kernel must be scalar or sgmma")
    request = dict(schema_version=1, model="qwen-image-2.1", operation="image.generate",
                prompt=PROMPT, output=str(output / (route["name"] + ".png")), inputs=[],
                width=512, height=512, steps=6, seed=42, frames=1, audio=False,
                execution="gpu", residency="component_staged", compile_gpu=False,
                # Both routes explicitly permit the pinned six-step distilled
                # student; approximate DiT caches remain separately disabled.
                allow_approximation=True, qwen21_dit_cache="off",
                qwen21_w8a8=False, prompt_enhance=False, qwen21_denoiser_backend=route["backend"],
                lora_strategy="inference_time",
                loras=[dict(path=str(adapter), strength=1.0, role="transformer")])
    # Preserve the historical scalar request and leave the MLX baseline alone.
    # This field selects a native implementation, not a numerical certificate.
    if route["name"] == "native" and native_lora_kernel == "sgmma":
        request["qwen21_native_lora_kernel"] = "sgmma"
    return request


def integer_field(value, expected, name):
    require(type(value) is int and value == expected, "unexpected actual " + name)


def object_field(value, key):
    result = value.get(key)
    require(isinstance(result, dict), "missing/malformed actual object: " + key)
    return result


def verify_result(result, route, request):
    for key, expected in (("actual_denoise_steps", 6), ("steps", 6), ("width", 512),
                          ("height", 512), ("seed", 42), ("reference_tokens", 0)):
        integer_field(result.get(key), expected, key)
    require(result.get("warmup") is False and result.get("prompt_cache_hit") is False,
            "fresh request must not be warmup or reuse conditioning")
    require(result.get("model") == request["model"] and result.get("operation") == "image.generate"
            and result.get("output") == request["output"], "actual model/operation/output mismatch")
    require(result.get("runtime_backend") == route["runtime_backend"]
            and result.get("lora_strategy") == "inference_time", "actual runtime LoRA executor mismatch")
    plan = object_field(result, "plan")
    require(plan.get("qwen21_dit_cache") == "off"
            and plan.get("algorithm_approximations") == [STUDENT_APPROXIMATION]
            and plan.get("execution") == "gpu", "actual plan changed the comparison recipe")
    timings = object_field(result, "timings_seconds")
    for key in ("request_wall", "denoise"):
        value = timings.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                "missing/invalid actual timing: " + key)
    if route["name"] == "native":
        # Runtime snapshot/import/run evidence, not a requested-plan echo.
        selected_kernel = request.get("qwen21_native_lora_kernel", "scalar")
        require(selected_kernel in LORA_RECIPES, "unknown requested native LoRA kernel")
        evidence = object_field(result, "native_denoiser_lora")
        require(evidence.get("enabled") is True and evidence.get("completed") is True
                and evidence.get("variant") == "viggle-r128-6step"
                and evidence.get("source_kind") == "verified-held-fd"
                and evidence.get("adapter_sha256") == ADAPTER_SHA256
                and evidence.get("precision_recipe") == LORA_RECIPES[selected_kernel],
                "native actual adapter pin/variant/math provenance mismatch")
        for key, expected in (("schema_version", 1), ("adapter_count", 1), ("resolved_bindings", 195),
                              ("logical_projections", 227), ("ordered_terms", 227),
                              ("max_rank", 128), ("completed_steps", 6)):
            integer_field(evidence.get(key), expected, key)
        for key in ("adapter_revision", "verified_artifact_digest"):
            require(isinstance(evidence.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", evidence[key]),
                    "native execution lacks actual " + key)
        for key in ("source_generation", "scratch_bytes"):
            require(type(evidence.get(key)) is int and evidence[key] > 0,
                    "native execution lacks actual " + key)
        require(type(evidence.get("source_strength")) in (int, float)
                and evidence["source_strength"] == 1.0, "native actual adapter strength differs")
        require(result.get("gpu_graph") == "native_full_dit_euler_runtime_lora",
                "native full DiT did not execute its runtime LoRA graph")
        require(result.get("runtime_precision") == "bf16_gpu+runtime_lora_fp32_rank",
                "native runtime LoRA did not use the intended FP32 rank recipe")
        phase = object_field(object_field(result, "native_denoiser_memory_phase"), "memory_admission")
        require(phase.get("execution_state") == "succeeded"
                and phase.get("worker_quarantined") is False and phase.get("tainted") is False,
                "native phase was not successfully released")
        for key in ZERO_LEDGER:
            integer_field(phase.get(key), 0, key)
        require(not result.get("native_denoiser_hybrid"), "LoRA comparison unexpectedly used hybrid")
    else:
        # Existing MLX loader verifies the r128 content pin at runtime. It
        # does not expose the native held-fd revision/phase proof; do not forge it.
        integer_field(result.get("lora_applied_projections"), 227, "MLX applied LoRA projections")
        require(not result.get("native_denoiser_lora"), "MLX baseline unexpectedly used the native adapter route")


def image_evidence(path):
    require(path.is_file() and not path.is_symlink(), "image must be a newly written regular PNG")
    with path.open("rb") as stream:
        header = stream.read(33)
    require(len(header) == 33 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[8:16] == b"\x00\x00\x00\rIHDR", "image has no valid PNG IHDR")
    require(zlib.crc32(header[12:29]) & 0xffffffff == struct.unpack(">I", header[29:33])[0],
            "PNG IHDR checksum mismatch")
    width, height = struct.unpack(">II", header[16:24])
    require((width, height) == (512, 512) and path.stat().st_size >= 45,
            "output PNG is not the expected 512x512 shape")
    return dict(path=str(path), bytes=path.stat().st_size, width=width, height=height,
                sha256=digest(path), validation="PNG header/shape only; no pixel-quality qualification")


def process_memory(log):
    text = log.read_text(errors="replace")
    result = {"scope": "whole CLI process time-l counters, not native ledger or MLX allocator"}
    for label, key in (("peak memory footprint", "peak_footprint_bytes"),
                       ("maximum resident set size", "max_rss_bytes"), ("swaps", "process_swaps")):
        values = re.findall(r"^\s*(\d+)\s+" + re.escape(label) + r"\s*$", text, re.MULTILINE)
        result[key] = int(values[-1]) if values else None
    return result


def write_json(path, value, exclusive=False):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(encoded)
    else:
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("x") as stream:
            stream.write(encoded)
        temporary.replace(path)


def run_pair(args):
    require(args.confirm_one_pair, "explicit --confirm-one-pair is required")
    selected_kernel = getattr(args, "native_lora_kernel", "scalar")
    require(selected_kernel in LORA_RECIPES, "native LoRA kernel must be scalar or sgmma")
    require(type(args.timeout_seconds) is int and 30 <= args.timeout_seconds <= 120,
            "timeout must be 30..120 seconds per route")
    native, model = args.native_dir.resolve(), args.model_root.resolve()
    # Keep the requested adapter basename; a model-library link is allowed.
    adapter, output = args.adapter.absolute(), args.output_dir.absolute()
    require(adapter.name == ADAPTER_NAME, "only the pinned local Viggle r128 adapter is permitted")
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir(),
            "output must be fresh in an existing parent; no overwrite/resume")
    before_source = source_metadata(model, adapter)
    build = build_metadata(native, args.expected_runtime_build_id)
    environment, removed = clean_environment()
    output.mkdir(exist_ok=False)
    receipt = dict(schema="qwen-native-runtime-lora-e2e-pair-v1", pair_complete=False,
                   performance_qualified=False, image_quality_verified=False,
                   full_request_memory_certified=False, model_downloads=0, checkpoint_copies=0,
                   samples_per_route=1, max_model_requests=2, timeout_seconds=args.timeout_seconds,
                   order=[route["name"] for route in ROUTES],
                   native_lora_kernel_requested=selected_kernel,
                   native_lora_recipe_expected=LORA_RECIPES[selected_kernel],
                   adapter_pin_expected=ADAPTER_SHA256,
                   approximation_scope="pinned six-step distillation only; approximate DiT cache and compilation disabled",
                   cache_state="fresh processes; shared OS/source/driver caches not reset; not cold-start proof",
                   source_proof_scope="supervisor stat only; MLX existing runtime pin check is pathname-based; native reports verified held-fd provenance",
                   source_before=before_source, build=build, environment_removed_keys=removed,
                   before=system_snapshot(), runs=[])
    receipt_path = output / "receipt.json"
    write_json(receipt_path, receipt, exclusive=True)
    for route in ROUTES:
        run = dict(name=route["name"], backend=route["backend"], status="not_started", termination_observed=False,
                   device_physical_completion_verified=False)
        receipt["runs"].append(run)
        try:
            require(source_metadata(model, adapter) == before_source, "sources changed before request admission")
            require(build_metadata(native, args.expected_runtime_build_id) == build,
                    "build artifacts changed before request admission")
            request = request_for(route, adapter, output, selected_kernel)
            name = route["name"]
            request_path = output / (name + "-request.json")
            stdout_path, stderr_path = output / (name + "-stdout.json"), output / (name + "-stderr.log")
            write_json(request_path, request, exclusive=True)
            command = ["/usr/bin/time", "-l", str(native / "turbocider"), "generate", str(model), str(request_path)]
            run.update(request=request, command=command, before=system_snapshot(),
                       stdout_path=str(stdout_path), stderr_path=str(stderr_path), status="running")
            write_json(receipt_path, receipt)
            print(f"start={name} timeout={args.timeout_seconds}s", flush=True)
            started = time.monotonic()
            with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                           env=environment, start_new_session=True)
                termination = wait_bounded(process, args.timeout_seconds)
            run.update(termination, supervised_wall_seconds=time.monotonic() - started,
                       after=system_snapshot(), process_memory=process_memory(stderr_path))
            require(not termination["timeout"] and not termination["cancelled"]
                    and termination["termination_observed"] and termination["process_returncode"] == 0,
                    "request failed/expired; process termination does not prove device completion")
            result = load_json(stdout_path)
            run["result"] = result
            verify_result(result, route, request)
            if name == "native":
                # Copy only successfully checked actual execution evidence.
                # The plan and request alone cannot populate this field.
                run["native_lora_kernel_requested"] = selected_kernel
                run["native_lora_actual_precision_recipe"] = result["native_denoiser_lora"]["precision_recipe"]
                receipt["native_lora_actual_precision_recipe"] = run["native_lora_actual_precision_recipe"]
            run["image"] = image_evidence(output / (name + ".png"))
            run["executed_route_and_adapter_verified"] = True
            run["native_phase_zero_ledger_verified"] = name == "native"
            # A completed native Session and zero phase provide its device
            # cleanup evidence; plain MLX output is not an equivalent ledger.
            run["device_physical_completion_verified"] = name == "native"
            run["status"] = "passed"
            print(f"done={name} wall={run['supervised_wall_seconds']:.3f}s", flush=True)
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
            run.update(status="failed", error=str(error))
        try:
            receipt["source_after"] = source_metadata(model, adapter)
            receipt["sources_unchanged"] = receipt["source_after"] == before_source
            receipt["build_after"] = build_metadata(native, args.expected_runtime_build_id)
            receipt["build_unchanged"] = receipt["build_after"] == build
        except (OSError, ValueError) as error:
            receipt.update(sources_unchanged=False, build_unchanged=False, verification_error=str(error))
        write_json(receipt_path, receipt)
        if run["status"] != "passed" or not receipt["sources_unchanged"] or not receipt["build_unchanged"]:
            receipt.update(pair_stopped=True, after=system_snapshot())
            write_json(receipt_path, receipt)
            print("pair_stopped=1; no further requests or automatic retry", flush=True)
            return 1
    receipt.update(pair_complete=True, after=system_snapshot())
    write_json(receipt_path, receipt)
    print(f"pair_complete=1 receipt={receipt_path}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runtime-build-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--native-lora-kernel", choices=tuple(LORA_RECIPES), default="scalar",
                        help="native GPU LoRA kernel; actual runtime recipe must match; MLX baseline is unchanged")
    parser.add_argument("--confirm-one-pair", action="store_true",
                        help="acknowledge exactly two bounded model requests, no retries")
    args = parser.parse_args(argv)
    try:
        return run_pair(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
