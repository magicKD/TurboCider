#!/usr/bin/env python3
"""Compare native T5 variants directly against captured Comfy conditioning."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-probe", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comfy-text", type=Path, required=True)
    parser.add_argument("--native-text", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-fp32", action="store_true", help="also enable FP32 T5 activations in the candidate")
    args = parser.parse_args()
    weights, tokenizer, tokenizer_probe, baseline, candidate, comfy_path, native_path = (p.resolve(strict=True) for p in
        (args.weights, args.tokenizer, args.tokenizer_probe, args.baseline, args.candidate, args.comfy_text, args.native_text))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file

    tokenize = subprocess.run([str(tokenizer_probe), str(tokenizer), "--t5"],
        input=json.dumps([{"prompt": args.prompt, "limit": 512}]), capture_output=True, text=True, timeout=60)
    if tokenize.returncode:
        raise RuntimeError(tokenize.stderr)
    tokens = json.loads(tokenize.stdout)[0]
    reference = load_file(str(comfy_path))
    valid = int(reference["attention_mask"].sum())
    if tokens["valid"] != valid or len(tokens["ids"]) != reference["conditioning"].shape[1]:
        raise ValueError("native and Comfy token geometry differs")
    output.mkdir(parents=True)
    input_path = output / "input.safetensors"
    save_file({"ids": torch.tensor(tokens["ids"], dtype=torch.int32),
               "valid": torch.tensor(valid, dtype=torch.int32)}, str(input_path))
    report = {"schema": "ltx098-t5-comfy-screen-v1", "status": "running", "valid_tokens": valid,
              "scope": "Real T5-XXL weights; exact prompt/token IDs; compare valid token embeddings to captured ComfyUI conditioning",
              "runs": []}
    try:
        candidate_options = ["--comfy-sdpa"] + (["--comfy-fp32"] if args.candidate_fp32 else [])
        for name, executable, extra in (("baseline", baseline, []), ("candidate", candidate, candidate_options)):
            result_path = output / f"{name}.safetensors"
            run = subprocess.run([str(executable), str(weights), str(input_path), str(result_path), "--t5-xxl", *extra],
                                 capture_output=True, text=True, timeout=600)
            if run.returncode:
                raise RuntimeError(f"{name}: {run.stderr[-4000:]}")
            result = load_file(str(result_path))["output"]
            expected = reference["conditioning"][:, :valid].float()
            actual = result[:, :valid].float()
            delta = actual - expected
            row = {"variant": name, "relative_l2": (delta.norm() / expected.norm()).item(),
                   "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
                   "max_abs": delta.abs().max().item()}
            if name == "baseline":
                original = load_file(str(native_path))["value"]
                row["matches_native_trace"] = bool(torch.equal(result, original))
                if not row["matches_native_trace"]:
                    raise ValueError("baseline probe differs from the captured native T5 output")
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
