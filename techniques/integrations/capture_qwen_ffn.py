#!/usr/bin/env python3
"""Capture nine original BF16 Qwen FFN inputs with the existing native hook.

Optional product integration, not part of the offline core. Uses local models
only, forced GPU FFNs, cache off, no LoRA or private backend. Capture timings are
instrumented and MUST NOT be reported as an ordinary request benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def source_match(build):
    inputs = build["inputs"]["sources"]
    for item in inputs:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError("native build source differs: " + item["path"])
    return len(inputs)


def stop_owned(process):
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError("owned capture process termination remains unconfirmed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--ane-manifest", type=Path, required=True,
                        help="explicit public runtime graph required by the product request contract; FFN chunks stay zero")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 60 <= args.timeout <= 300:
        parser.error("capture timeout must be 60..300 seconds")
    output, native, model = [path.resolve() for path in (args.output_dir, args.native_dir, args.model_root)]
    try:
        build = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
        source_count = source_match(build)
        manifest = json.loads(args.ane_manifest.read_text())
        if any(manifest.get(key) != value for key, value in
               {"backend": "runtime_weight_fp16", "kind": "swiglu", "rows": 1024, "hidden": 4096, "width": 12288}.items()):
            raise ValueError("capture requires the public full-width 1024x4096x12288 runtime SwiGLU manifest")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    cli, library = native / "turbocider", native / "libturbocider.dylib"
    checkpoint = model / "diffusion_models/qwen_image_2.1_bf16.safetensors"
    before = checkpoint.stat()
    output.mkdir(parents=True, exist_ok=False)
    request = dict(model="qwen-image-2.1", operation="image.generate", width=512, height=512,
                   prompt="A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
                   output=str(output / "capture.png"), steps=20, seed=args.seed, frames=1, audio=False,
                   execution="gpu_ane", residency="resident", allow_approximation=True,
                   hybrid_mlp_mode="runtime", ane_manifest=str(args.ane_manifest.resolve()),
                   qwen21_dit_cache="off", qwen21_denoiser_backend="mlx")
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("TURBOCIDER_", "ANE_", "ANEFORGE_"))}
    environment.update(TURBOCIDER_ANE_CALIBRATION_DIR=str(output / "captures"),
                       TURBOCIDER_ANE_CALIBRATION_REPLAY="1", TURBOCIDER_RUNTIME_ANE_CHUNKS="0",
                       TURBOCIDER_RUNTIME_ANE_PROFILE="0", TURBOCIDER_ANE_BACKEND="public")
    report = dict(schema="tc-qwen-real-ffn-capture-v1", status="running", not_performance=True,
                  production_qualified=False, runtime_build_id=build["runtime_build_id"],
                  cli_sha256=digest(cli), library_sha256=digest(library), native_source_files_matched=source_count,
                  runner_sha256=digest(Path(__file__)), seed=args.seed, steps=20, loras=[],
                  ane_manifest_sha256=digest(args.ane_manifest),
                  shape=[1, 1024, 4096], original_dtype="bfloat16", cache="off",
                  requested_ffn_route="forced-gpu-runtime-boundary", actual_ffn_route="not-confirmed", records=[],
                  checkpoint_stat_before={key: getattr(before, key) for key in ("st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev")})
    receipt = output / "receipt.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2) + "\n")

    save()
    started = time.monotonic()
    process = None
    try:
        with (output / "stdout.json").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
            process = subprocess.Popen([str(cli), "generate", str(model), str(request_path)],
                                       env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            report["owned_pid"] = process.pid
            save()
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                report["returncode"] = stop_owned(process)
                raise RuntimeError("capture exceeded its time budget; owned process stopped")
        report["returncode"] = code
        if code:
            raise RuntimeError("native capture failed; inspect its local stderr log")
        source_match(build)
        after = checkpoint.stat()
        if any(getattr(before, key) != getattr(after, key) for key in report["checkpoint_stat_before"]):
            raise ValueError("checkpoint stat identity changed")
        completions = list((output / "captures").glob("*-replay/complete.json"))
        if len(completions) != 1:
            raise ValueError("capture did not publish exactly one replay completion receipt")
        directory = completions[0].parent
        wanted = {f"layer-{layer}-step-{step}.safetensors" for layer in (0, 15, 31) for step in (1, 10, 19)}
        files = list(directory.glob("*.safetensors"))
        if {path.name for path in files} != wanted or sum(path.stat().st_size for path in files) > 73 * 1024 * 1024:
            raise ValueError("capture coverage or byte bound differs")
        report["replay_directory"] = str(directory.relative_to(output))
        report["completion_sha256"] = digest(completions[0])
        report["records"] = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                               "sha256": digest(path)} for path in sorted(files)]
        report["status"] = "captured"
        report["actual_ffn_route"] = "forced-gpu-runtime-boundary"
    except Exception as error:
        if process is not None and process.poll() is None:
            report["returncode"] = stop_owned(process)
        report.update(status="failed", error=str(error))
    report["instrumented_wall_seconds"] = time.monotonic() - started
    save()
    print(json.dumps(report, indent=2))
    return int(report["status"] != "captured")


if __name__ == "__main__":
    raise SystemExit(main())
