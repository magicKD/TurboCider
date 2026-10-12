"""Plan a local Qwen edit; --run submits it and an optional second edit.

Existing weights only. No downloads. Inputs keep their given order. The second
stage uses the first result as its sole reference and starts only on success.
"""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings/python"))
from turbocider_local import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="Socket path shown by the App")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--turbo-lora", required=True, type=Path, help="Existing Viggle six-step adapter")
    parser.add_argument("--reference", type=Path, action="append", required=True, help="Repeat for 1–3 ordered images")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--then", help="Optional second editing prompt, using the first result")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-size", type=int, choices=(1024, 512), default=1024,
                        help="Reference encoding area: 1024 standard; 512 faster preview (r128 only, may lose detail). Output stays 512×512.")
    parser.add_argument("--run", action="store_true", help="Submit after planning; otherwise only inspect the first plan")
    args = parser.parse_args()
    if not 1 <= len(args.reference) <= 3:
        parser.error("Qwen Turbo supports 1–3 reference images in this example")
    if not args.model.is_dir() or not args.turbo_lora.is_file() or not all(p.is_file() for p in args.reference):
        parser.error("Model, adapter and reference files must already exist locally")
    client = Client(args.socket)
    print(json.dumps(client.capabilities(), ensure_ascii=False, indent=2), flush=True)
    run_id = uuid.uuid4().hex
    references = args.reference
    prompts = [args.prompt] + ([args.then] if args.then else [])
    for index, prompt in enumerate(prompts):
        output = args.output_dir.resolve() / f"{run_id}-{index + 1}.png"
        request = {
            "schema_version": 1, "model": "qwen-image-2.1", "operation": "image.edit",
            "prompt": prompt, "inputs": [{"kind": "image", "role": "reference", "path": str(p.resolve())} for p in references],
            "output": str(output), "width": 512, "height": 512, "steps": 6, "seed": args.seed,
            "execution": "gpu", "residency": "component_staged", "dynamic_text": True,
            "prompt_enhance": False, "allow_approximation": True, "qwen21_dit_cache": "off",
            "qwen21_reference_size": args.reference_size,
            "lora_strategy": "inference_time",
            "loras": [{"path": str(args.turbo_lora.resolve()), "role": "transformer", "strength": 1.0}],
        }
        print(json.dumps({"stage": index + 1, "plan": client.plan(request)}, ensure_ascii=False, indent=2), flush=True)
        if not args.run:
            print("Plan only. Add --run to generate; the next stage is planned after its reference exists.")
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        job_id = client.submit(str(args.model.resolve()), request)
        print(json.dumps({"stage": index + 1, "job_id": job_id, "output": str(output)}), flush=True)
        job = client.wait(job_id)
        print(json.dumps(job, ensure_ascii=False, indent=2), flush=True)
        if not output.is_file():
            raise RuntimeError(f"Successful job did not create the expected stage output: {output}")
        references = [output]


if __name__ == "__main__":
    main()
