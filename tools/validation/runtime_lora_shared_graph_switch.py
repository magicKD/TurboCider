#!/usr/bin/env python3
"""Exercise one frozen Qwen/Z-Image Core ML graph across two runtime adapters.

The second adapter is deliberately synthetic: it verifies session/identity
re-binding, not its learned image quality. Temporary adapter and PNGs are
removed when the check exits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/native/turbocider")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-request", type=Path, required=True)
    parser.add_argument("--base-request", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    adapter_request = json.loads(args.adapter_request.read_text())
    base_request = json.loads(args.base_request.read_text())
    model_id = adapter_request["model"]
    require(model_id == base_request["model"] and
            model_id in ("z-image-turbo", "qwen-image-2.1"),
            "requests must target the same supported frozen-base model")
    expected_steps = 6 if model_id == "qwen-image-2.1" else 8
    for request in (adapter_request, base_request):
        require(request.get("schema_version", 1) == 1 and
                request.get("width") == request.get("height") == 512 and
                request.get("steps") == expected_steps and
                request.get("execution") == "gpu_ane" and
                request.get("residency") == "resident" and
                request.get("allow_approximation") is True,
                "switch probe requires matched explicit resident 512px requests")
    require(adapter_request["hybrid_mlp_mode"] == base_request["hybrid_mlp_mode"] == "lora_fused" and
            adapter_request["ane_manifest"] == base_request["ane_manifest"],
            "requests must share one frozen-base fused Core ML manifest")
    require(len(adapter_request.get("loras", [])) == 1 and not base_request.get("loras"),
            "provide one original runtime adapter and one base-only request")
    require(adapter_request.get("lora_strategy") == "inference_time" and
            adapter_request["loras"][0].get("role") == "transformer",
            "the adapter request must use inference-time transformer LoRA")
    adapter = (ROOT / adapter_request["loras"][0]["path"]).resolve()
    require(adapter.is_file(), "original adapter is missing")
    with tempfile.TemporaryDirectory(prefix="turbocider-lora-switch-") as directory:
        temporary = Path(directory)
        synthetic = temporary / "synthetic-different-transformer-lora.safetensors"
        arrays = mx.load(str(adapter))
        changed = 0
        for key, value in arrays.items():
            if ".lora_B." in key:
                arrays[key] = mx.astype(value * 0.5, value.dtype)
                changed += 1
        require(changed > 0, "the source adapter has no LoRA B tensors")
        mx.save_safetensors(str(synthetic), arrays)
        del arrays
        mx.clear_cache()
        require(sha256(synthetic) != sha256(adapter), "synthetic adapter did not change")
        request_files = []
        for index, (prototype, active_adapter) in enumerate((
            (base_request, None), (adapter_request, adapter),
            (adapter_request, synthetic), (base_request, None)
        )):
            request = json.loads(json.dumps(prototype))
            request["output"] = str(temporary / f"image-{index}.png")
            if active_adapter:
                request["loras"][0]["path"] = str(active_adapter)
            target = temporary / f"request-{index}.json"
            target.write_text(json.dumps(request))
            request_files.append(target)
        command = [str(args.cli.resolve()), "batch", str(args.model.resolve()),
                   *(str(path) for path in request_files)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=args.timeout, env=os.environ.copy())
        require(completed.returncode == 0, completed.stderr[-3000:])
        rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        require(len(rows) == 4, "expected four successful resident requests")
        calls = [row["hybrid"]["runtime_calls_session_total"] for row in rows]
        loads = [row["hybrid"]["load_seconds"] for row in rows]
        applied = [row.get("lora_applied_projections", 0) for row in rows]
        outputs = [sha256(temporary / f"image-{index}.png") for index in range(4)]
        calls_per_request = 160 if model_id == "qwen-image-2.1" else 256
        expected_projections = 227 if model_id == "qwen-image-2.1" else 238
        require(calls == [calls_per_request * n for n in range(1, 5)],
                "Core ML session did not persist")
        require(loads.count(loads[0]) == 4, "Core ML model was reloaded")
        require(applied == [0, expected_projections, expected_projections, 0],
                "adapter was not rebound cleanly")
        require(outputs[0] == outputs[3], "adapter contaminated the returned base request")
        require(outputs[1] != outputs[2], "the two distinct adapters generated identical PNGs")
        print(json.dumps({"model": model_id,
                          "source_adapter_sha256": sha256(adapter),
                          "synthetic_adapter_sha256": sha256(synthetic),
                          "synthetic_lora_b_tensors": changed,
                          "coreml_calls": calls, "bound_adapter_projections": applied,
                          "base_sha256": outputs[0], "adapter_png_sha256": outputs[1:3]},
                         sort_keys=True))


if __name__ == "__main__":
    main()
