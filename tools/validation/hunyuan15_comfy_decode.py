#!/usr/bin/env python3
"""Decode a saved Hunyuan 1.5 latent in a fresh, isolated ComfyUI process.

This recovery path measures only VAE + MP4 export, not complete generation.
It avoids keeping the text encoder and Transformer alive during decode.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from hunyuan15_comfy_reference import COMPONENTS, run_request
from ltx098_comfy_reference import sha256
from z_comfy_framework_screen import launch_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "comfy", "extra", "model", "latent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()
    import yaml
    if args.output.exists():
        parser.error("output directory must be new")
    if args.fps <= 0 or args.tile_size < 128 or args.tile_size % 32:
        parser.error("invalid fps or tile size")
    model, comfy, extra, latent = (path.resolve(strict=True) for path in (args.model, args.comfy, args.extra, args.latent))
    directory, filename, expected = COMPONENTS["vae"]
    actual = sha256((model / directory / filename).resolve(strict=True))
    if actual != expected:
        raise ValueError(f"VAE checksum mismatch: {actual}")
    output = args.output.absolute()
    output.mkdir(parents=True)
    instance = output / "instance"
    (instance / "input").mkdir(parents=True)
    (instance / "user").mkdir()
    shutil.copyfile(latent, instance / "input/input.latent")
    (output / "models.yaml").write_text(yaml.safe_dump({"hunyuan15": {"base_path": str(model), "vae": "vae"}}))
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
        "1": node("LoadLatent", latent="input.latent"),
        "5": node("VAELoader", vae_name=filename),
        "12": node("VAEDecodeTiled", samples=["1", 0], vae=["5", 0], tile_size=args.tile_size,
                   overlap=32 if args.tile_size < 256 else 64, temporal_size=4096, temporal_overlap=8),
        "13": node("CreateVideo", images=["12", 0], fps=float(args.fps), bit_depth=8),
        "14": node("SaveVideo", video=["13", 0], filename_prefix="hunyuan15-recovered",
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }
    (output / "workflow.json").write_text(json.dumps(graph, indent=2) + "\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = launch_command(comfy, args.python.absolute(), output, "comfy-sdpa-gpu", port)
    command.remove("--use-pytorch-cross-attention")
    command.append("--use-quad-cross-attention")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    report = {"schema": "hunyuan15-comfy-decode-v1", "status": "running",
              "scope": "separate VAE-only process; decode/export timing is not full generation",
              "source_latent_sha256": sha256(latent), "vae_sha256": actual,
              "fps": args.fps, "tile_size": args.tile_size, "temporal_size": 4096}
    try:
        with (output / "server.stdout.txt").open("x") as stdout, (output / "server.stderr.txt").open("x") as stderr:
            server = subprocess.Popen(command, cwd=comfy, env=environment, stdout=stdout, stderr=stderr)
            try:
                result = asyncio.run(run_request(server, port, graph, output, None, ("1", "5", "12", "14")))
                report["result"] = result
                report["status"] = "complete"
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)
                report["server_returncode"] = server.returncode
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "decode_seconds": report["result"]["request_wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
