#!/usr/bin/env python3
"""Exercise one Qwen/Z-Image Core ML graph across two runtime adapters.

The second adapter is deliberately synthetic: it verifies session/identity
re-binding, not its learned image quality. Supports frozen lora_fused and
runtime-weight activation-input graphs. The synthetic adapter is temporary;
--output optionally preserves images, requests and raw telemetry.
Model diagnostic environment overrides are cleared for this state-isolation test.
"""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from runtime_ane_common import (
    benchmark_environment, session_counter, validate_edit_results, validate_results, wait_for_idle,
    qwen_qk_environment, validate_qwen_qk_receipts,
)
# Preserve the helper name used by local evidence-analysis scripts.
from runtime_ane_common import sha256_file as sha256


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_requests(base_request, adapter_request):
    """Validate the state-test workload before importing MLX or writing evidence.

    Frozen fixed call counts cover generation. Runtime also accepts Qwen
    ref512 editing, and requires actual predictions on every request.
    Only adapter binding and output location may differ between prototypes.
    Keep native request validation as the authority for model/artifact support.
    """
    require(isinstance(base_request, dict) and isinstance(adapter_request, dict),
            "requests must be JSON objects")
    model_id = adapter_request.get("model")
    require(model_id == base_request.get("model") and
            model_id in ("z-image-turbo", "qwen-image-2.1"),
            "requests must target the same supported model")
    expected_steps = 6 if model_id == "qwen-image-2.1" else 8
    for request in (adapter_request, base_request):
        require(request.get("schema_version", 1) == 1 and
                request.get("width") == request.get("height") == 512 and
                request.get("steps") == expected_steps and
                request.get("execution") == "gpu_ane" and
                request.get("residency") == "resident" and
                request.get("allow_approximation") is True,
                "switch probe requires matched explicit resident 512px requests")
        if request.get("operation") == "image.edit":
            references = request.get("inputs")
            require(model_id == "qwen-image-2.1" and
                    request.get("hybrid_mlp_mode") == "runtime" and
                    type(request.get("qwen21_reference_size")) is int and
                    request.get("qwen21_reference_size") == 512 and
                    isinstance(references, list) and 1 <= len(references) <= 3 and
                    all(isinstance(item, dict) and item.get("kind") == "image" and
                        item.get("role") == "reference" and
                        isinstance(item.get("path"), str) and item["path"].strip()
                        for item in references),
                    "editing switch requires Qwen runtime and 1...3 ordered ref512 images")
        else:
            require(request.get("operation") == "image.generate" and not request.get("inputs"),
                    "switch probe requires generation or supported Qwen runtime editing")
        require(type(request.get("seed")) is int and 0 <= request["seed"] <= 2147483647,
                "switch probe requires an explicit deterministic seed")
        require(isinstance(request.get("prompt"), str) and request["prompt"].strip(),
                "switch probe requires a nonempty prompt")
    mode = adapter_request.get("hybrid_mlp_mode")
    manifest = adapter_request.get("ane_manifest")
    require(mode in ("lora_fused", "runtime") and mode == base_request.get("hybrid_mlp_mode") and
            isinstance(manifest, str) and manifest.strip() and
            manifest == base_request.get("ane_manifest"),
            "requests must share one complete base/activation-input Core ML manifest")
    adapters = adapter_request.get("loras")
    require(isinstance(adapters, list) and len(adapters) == 1 and
            isinstance(adapters[0], dict) and not base_request.get("loras"),
            "provide one original runtime adapter and one base-only request")
    require(adapter_request.get("lora_strategy") == "inference_time" and
            adapters[0].get("role") == "transformer" and
            isinstance(adapters[0].get("path"), str) and adapters[0]["path"].strip(),
            "the adapter request must use inference-time transformer LoRA")
    ignored = {"output", "loras", "lora_strategy"}

    def workload(request):
        return {"schema_version": 1,
                **{key: value for key, value in request.items() if key not in ignored}}

    require(workload(base_request) == workload(adapter_request),
            "base and adapter workloads must match except output, loras and lora_strategy")
    return model_id, mode


def reference_identity(request):
    """Ordered pre/post byte identity, not an immutable artifact lease."""
    return [{"path": str((ROOT / item["path"]).resolve()),
             "sha256": sha256(ROOT / item["path"])} for item in request.get("inputs", [])]


def validate_edit_receipts(rows, request):
    """Keep the switch helper API while sharing the screen's receipt checks."""
    if request["operation"] != "image.edit":
        return None
    try:
        return validate_edit_results(rows, request)
    except ValueError as error:
        raise AssertionError(str(error)) from error


def validate_prediction_progress(calls):
    # Especially important when chunk > 512px decode length: a route label
    # alone cannot qualify adapter corrections or graph reuse on Core ML.
    require(calls and all(type(value) is int and value >= 0 for value in calls) and
            all(later > earlier for earlier, later in zip([0] + calls, calls)),
            "runtime-weight switch requires actual predictions on every request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/native/turbocider")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-request", type=Path, required=True)
    parser.add_argument("--base-request", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, help="new evidence directory; synthetic adapter remains temporary")
    parser.add_argument("--qwen-qk-norm-rope", action="store_true",
                        help="explicit fused GPU Q/K norm-RoPE for Qwen 512px state-isolation checks")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    try:
        adapter_request = json.loads(args.adapter_request.read_text())
        base_request = json.loads(args.base_request.read_text())
        model_id, mode = validate_requests(base_request, adapter_request)
        qk_environment = qwen_qk_environment(model_id, base_request["width"], args.qwen_qk_norm_rope)
        references = reference_identity(base_request)
    except (OSError, ValueError, AssertionError) as error:
        parser.error(str(error))
    adapter = (ROOT / adapter_request["loras"][0]["path"]).resolve()
    require(adapter.is_file(), "original adapter is missing")
    identity_paths = {"source_adapter_sha256": adapter,
                      "manifest_sha256": ROOT / base_request["ane_manifest"],
                      "library_sha256": args.cli.resolve().parent / "libturbocider.dylib"}
    identities = {key: sha256(path) for key, path in identity_paths.items()}
    # CLI help and host contracts do not need the optional MLX dependency.
    import mlx.core as mx

    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    # Adapter synthesis itself uses MLX; wait before it, not just before CLI inference.
    snapshot = wait_for_idle()
    with tempfile.TemporaryDirectory(prefix="turbocider-lora-switch-") as directory:
        temporary = Path(directory)
        evidence = args.output.resolve() if args.output else temporary
        (evidence / "preparation-processes.txt").write_text(snapshot)
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
        sequence = [
            (base_request, None), (adapter_request, adapter),
            (adapter_request, synthetic), (base_request, None)
        ]
        if mode == "runtime":
            # Discover the base's FP16 headroom before saving the reference
            # image. Pin chunk count so this checks state, not timing choices.
            sequence.insert(0, (base_request, None))
        for index, (prototype, active_adapter) in enumerate(sequence):
            request = json.loads(json.dumps(prototype))
            request["output"] = str(evidence / f"image-{index}.png")
            if active_adapter:
                request["loras"][0]["path"] = str(active_adapter)
            target = evidence / f"request-{index}.json"
            target.write_text(json.dumps(request))
            request_files.append(target)
        command = [str(args.cli.resolve()), "batch", str(args.model.resolve()),
                   *(str(path) for path in request_files)]
        snapshot = wait_for_idle()
        (evidence / "processes.txt").write_text(snapshot)
        env = benchmark_environment()
        env.update(qk_environment)
        if mode == "runtime":
            env["TURBOCIDER_RUNTIME_ANE_CHUNKS"] = "1"
        if base_request["operation"] == "image.edit":
            env["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"] = "1"
        require(reference_identity(base_request) == references,
                "reference images changed before switch")
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=args.timeout, env=env)
        (evidence / "stdout.jsonl").write_text(completed.stdout)
        (evidence / "stderr.txt").write_text(completed.stderr)
        require(completed.returncode == 0, completed.stderr[-3000:])
        require(reference_identity(base_request) == references,
                "reference images changed during switch")
        require({key: sha256(path) for key, path in identity_paths.items()} == identities,
                "base graph, runtime library or source adapter changed during switch")
        rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        require(len(rows) == len(sequence), "wrong resident request count")
        route = "runtime" if mode == "runtime" else "frozen"
        validate_results(rows, route, len(sequence), model_id)
        if model_id == "qwen-image-2.1":
            validate_qwen_qk_receipts(rows, args.qwen_qk_norm_rope)
        # Validate adapter rows separately; unbound or partial LoRA is not a
        # passing switch even when it produces a PNG.
        offset = 1 if mode == "runtime" else 0
        validate_results(rows[offset + 1:offset + 3], route, 2, model_id, expect_lora=True)
        reference_tokens = validate_edit_receipts(rows, base_request)
        calls = [row["hybrid"]["runtime_calls_session_total"] for row in rows]
        loads = [row["hybrid"]["load_seconds"] for row in rows]
        applied = [row.get("lora_applied_projections", 0) for row in rows]
        outputs = [sha256(evidence / f"image-{index}.png") for index in range(len(sequence))]
        calls_per_request = 160 if model_id == "qwen-image-2.1" else 256
        expected_projections = 227 if model_id == "qwen-image-2.1" else 238
        if mode == "runtime":
            validate_prediction_progress(calls)
        else:
            require(calls == [calls_per_request * n for n in range(1, 5)],
                    "Core ML session did not persist")
        require(loads.count(loads[0]) == len(sequence), "Core ML model was reloaded")
        require(applied == ([0] if offset else []) + [0, expected_projections, expected_projections, 0],
                "adapter was not rebound cleanly")
        require(outputs[offset] == outputs[-1], "adapter contaminated the returned base request")
        require(outputs[offset + 1] != outputs[offset + 2], "the two distinct adapters generated identical PNGs")
        summary = {
            "model": model_id, "mode": mode,
            "qwen_qk_norm_rope": args.qwen_qk_norm_rope,
            "operation": base_request["operation"],
            "references": references, "reference_tokens": reference_tokens,
            **identities,
            "synthetic_adapter_sha256": sha256(synthetic),
            "synthetic_lora_b_tensors": changed,
            "coreml_calls": calls, "bound_adapter_projections": applied,
            "base_sha256": outputs[offset],
            "adapter_png_sha256": outputs[offset + 1:offset + 3],
        }
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
