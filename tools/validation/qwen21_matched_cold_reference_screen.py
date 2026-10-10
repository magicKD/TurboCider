#!/usr/bin/env python3
"""Serial actual BF16-I/O-preconditioned native/reference cold+warm comparison."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys

from qwen21_bf16_stream_screen import validate_bf16_rows, validate_bf16_registration
from qwen21_gguf_hybrid_screen import PROMPTS, validate_receipts, validate_verified_imports
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
MODES = ("gpu", "reference", "mixed")


def file_identity(path):
    try:
        target = path.resolve(strict=True)
        st = path.stat()
    except OSError as error:
        raise ValueError("original source path is unavailable") from error
    if not path.is_file() or st.st_size <= 0:
        raise ValueError("missing regular original source")
    return dict(target=str(target), device=st.st_dev, inode=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def check_identities(identities):
    if any(file_identity(Path(name)) != identity for name, identity in identities.items()):
        raise ValueError("original source generation changed during conditioned window")


def native_env(cache, *, streamed=False, mixed=False):
    env = benchmark_environment()
    env.update(TURBOCIDER_QWEN21_SOURCE_REGISTRATION="1", TURBOCIDER_SOURCE_REGISTRATION_CACHE=str(cache),
               TURBOCIDER_QWEN21_PROFILE_STEPS="1", TURBOCIDER_QWEN21_BF16_STREAMING="1" if streamed else "0",
               TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS="0" if streamed else "1",
               TURBOCIDER_QWEN21_BF16_STREAM_ENCODER_PREFIX="28", TURBOCIDER_QWEN21_BF16_STREAM_DIT_PREFIX="28",
               TURBOCIDER_QWEN21_BF16_STREAM_WEIGHT_GIB="13", TURBOCIDER_QWEN21_GGUF_DECODE_WORKERS="8",
               TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU="0" if streamed else "1",
               TURBOCIDER_QWEN21_GGUF_VERIFY_IMPORT_OVERLAP="0" if streamed else "1",
               TURBOCIDER_QWEN21_GGUF_READ_BUFFER_MIB="1" if streamed else "4",
               TURBOCIDER_QWEN21_GGUF_READ_AHEAD="0" if streamed else "1")
    if mixed:
        env.update(TURBOCIDER_ANE_BACKEND="private", TURBOCIDER_ALLOW_PRIVATE_ANE="1",
            TURBOCIDER_PRIVATE_ANE_CHANNELS="5120", TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",
            TURBOCIDER_PRIVATE_ANE_GPU_IO="1", TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",
            TURBOCIDER_RUNTIME_ANE_CHUNKS="1", TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE="1",
            TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1", TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="0",
            TURBOCIDER_PRIVATE_ANE_PREFETCH="0", TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD="0",
            TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE="all")
    return env


def write_requests(output, prefix, steps, *, streamed=False, mixed=False, manifest=None):
    paths = []
    for index, prompt in enumerate(PROMPTS):
        request = dict(model="qwen-image-2.1", operation="image.generate", prompt=prompt, width=512, height=512,
                       steps=steps, seed=29, audio=False, execution="gpu_ane" if mixed else "gpu",
                       residency="component_staged" if streamed else "resident", allow_approximation=True,
                       output=str(output / f"{prefix}-{index}.png"))
        if mixed:
            request.update(hybrid_mlp_mode="runtime", ane_manifest=str(manifest))
        path = output / f"{prefix}-{index}.json"
        path.write_text(json.dumps(request, indent=2) + "\n")
        paths.append(str(path))
    return paths


def run_native(cli, model, requests, output, stem, env):
    with (output / f"{stem}.stdout.jsonl").open("x") as stdout, (output / f"{stem}.stderr.txt").open("x") as stderr:
        memory = run_sampled([str(cli), "batch", str(model), *requests], repo=ROOT, output=output, stem=stem,
                             env=env, stdout=stdout, stderr=stderr, timeout=900, interval_ms=100, max_gap_ms=500)
    rows = [json.loads(line) for line in (output / f"{stem}.stdout.jsonl").read_text().splitlines()]
    return rows, memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registration-cache", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 40), default=40)
    parser.add_argument("--order", default=",".join(MODES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    order = args.order.split(",")
    if len(order) != len(set(order)) or set(order) != set(MODES) or args.output.exists() or args.output.is_symlink():
        parser.error("all three serial arms and fresh evidence required")
    cli, reference, binary, bf16, gguf, manifest, cache = (p.resolve(strict=True) for p in
        (args.cli, args.reference, args.binary, args.bf16, args.gguf, args.manifest, args.registration_cache))
    # Read-only reference imports; do not create pyc files in another repo.
    sys.dont_write_bytecode = True
    backend = reference / "studio/backend"
    sys.path.insert(0, str(backend))
    from core.inference.sd_cpp_args import metal_text_encoder_flags, native_speed_flags
    placement = "UNSLOTH_DIFFUSION_SD_CPP_METAL_TE_GPU"
    previous = os.environ.get(placement)
    try:
        os.environ[placement] = "1"
        flags = metal_text_encoder_flags() + native_speed_flags("max")
    finally:
        if previous is None: os.environ.pop(placement, None)
        else: os.environ[placement] = previous
    if "--clip-on-cpu" in flags:
        raise ValueError("original reference GPU conditioner flags differ")
    encoder = gguf / "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
    dit = gguf / "diffusion_models/qwen-image-2.1-Q4_K_M.gguf"
    vae = gguf / "vae/qwen_image_2.1_vae_bf16.safetensors"
    bf_encoder = bf16 / "text_encoders/qwen3vl_8b_bf16.safetensors"
    bf_dit = bf16 / "diffusion_models/qwen_image_2.1_bf16.safetensors"
    processor = gguf / "processor/tokenizer.json"
    worker_script = ROOT / "tools/validation/qwen21_gguf_unsloth_encoder_screen.py"
    sources = (cli, cli.parent / "libturbocider.dylib", binary, manifest, encoder, dit, vae, bf_encoder, bf_dit,
               processor, worker_script, backend / "core/inference/sd_cpp_args.py", backend / "core/inference/diffusion_memory.py")
    identities = {str(path): file_identity(path) for path in sources}
    # All full source hashes BEFORE conditioning; only stat/canonical checks
    # between conditioning and target launch, never a GGUF warming hash pass.
    hashes = {str(path): sha256_file(path) for path in sources}
    check_identities(identities)
    output = args.output.absolute(); output.mkdir(parents=True)
    summary = dict(schema="tc-qwen21-bf16-conditioned-reference-window-v1", status="running", qualification_passed=False,
                   steps=args.steps, precondition_steps=40, order=order, source_identities=hashes, file_generations=identities,
                   scope="same actual BF1640 x3 precondition per arm; no post-precondition payload hash/cache purge; not identical physical cache/engine-noise proof",
                   trials=[])
    destination = output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for mode in order:
        print(json.dumps(dict(starting=mode, phase="bf16_precondition")), flush=True)
        check_identities(identities)
        requests = write_requests(output, mode + "-precondition", 40, streamed=True)
        before, pre_memory = run_native(cli, bf16, requests, output, mode + "-precondition",
                                       native_env(cache / "bf16-stream", streamed=True))
        validate_bf16_rows(before, 40, True, (28, 28), 13 << 30,
                           dict(encoder=hashes[str(bf_encoder)], denoiser=hashes[str(bf_dit)]), "reuse")
        logs = (output / f"{mode}-precondition.stderr.txt").read_text().splitlines()
        registrations = [json.loads(line)["qwen_source_registration"] for line in logs
                         if line.startswith('{"qwen_source_registration":')]
        validate_bf16_registration(registrations, "reuse", {bf_encoder.name: bf_encoder.stat().st_size, bf_dit.name: bf_dit.stat().st_size})
        check_identities(identities)
        print(json.dumps(dict(starting=mode, phase="target")), flush=True)
        if mode == "reference":
            target = output / "reference"; target.mkdir()
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0)); port = reservation.getsockname()[1]
            command = [str(binary), "--diffusion-model", str(dit), "--llm", str(encoder), "--vae", str(vae),
                       *flags, "--threads", "12", "--verbose", "--cfg-scale", "1", "--listen-ip", "127.0.0.1", "--listen-port", str(port)]
            (target / "launch.json").write_text(json.dumps(dict(command=command, port=port, steps=args.steps), indent=2) + "\n")
            with (target / "driver.stdout.jsonl").open("x") as stdout, (target / "driver.stderr.txt").open("x") as stderr:
                memory = run_sampled([sys.executable, str(worker_script), "--worker", "--output", str(target)],
                    repo=ROOT, output=target, stem="reference", env=benchmark_environment(), stdout=stdout, stderr=stderr,
                    timeout=900, interval_ms=100, max_gap_ms=500, child_roles=(("reference_inference", str(binary)),))
            result = json.loads((target / "conditions.json").read_text())
            log = (target / "server.stdout.txt").read_text() + (target / "server.stderr.txt").read_text()
            if len(result["requests"]) != 3 or not re.search(r"Conditioner[^\n]*-> compute MTL0, params MTL0", log):
                raise ValueError("original GPU reference placement/lifecycle missing")
            target_result = dict(reference=result, command=command, warm_medians={field: statistics.median(row[field]
                for row in result["requests"][1:]) for field in
                ("condition_seconds", "sampling_seconds", "generation_seconds", "http_request_seconds")})
        else:
            mixed = mode == "mixed"
            requests = write_requests(output, mode, args.steps, mixed=mixed, manifest=manifest)
            rows, memory = run_native(cli, gguf, requests, output, mode,
                native_env(cache / ("gguf-ane" if mixed else "gguf-gpu"), mixed=mixed))
            validate_receipts(rows, 5120 if mixed else 0, args.steps, "all", True)
            imports = [json.loads(line)["qwen_gguf_verified_import"] for line in
                (output / f"{mode}.stderr.txt").read_text().splitlines() if line.startswith('{"qwen_gguf_verified_import":')]
            validate_verified_imports(imports, dict(encoder=(hashes[str(encoder)], encoder.stat().st_size),
                                                   dit=(hashes[str(dit)], dit.stat().st_size)), "reuse", 4, True)
            if any(row["qwen_encoder_prefill"]["processor_sha256"] != hashes[str(processor)] for row in rows):
                raise ValueError("native original processor differs")
            target_result = dict(timings=[row["timings_seconds"] for row in rows], verified_imports=imports,
                warm_medians={field: statistics.median(row["timings_seconds"][field] for row in rows[1:])
                              for field in ("request_wall", "text_encode", "denoise", "vae_decode")},
                png_sha256=[sha256_file(output / f"{mode}-{index}.png") for index in range(3)])
        check_identities(identities)
        summary["trials"].append(dict(mode=mode, precondition=dict(timings=[row["timings_seconds"] for row in before],
            source_registrations=registrations, memory=pre_memory), target=target_result, memory=memory))
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    if any(sha256_file(path) != hashes[str(path)] for path in sources):
        raise ValueError("original source/reference/runtime changed during window")
    check_identities(identities)
    summary["status"] = "complete_diagnostic"
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
