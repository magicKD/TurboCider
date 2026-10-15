#!/usr/bin/env python3
"""Opt-in end-to-end ComfyUI test using existing registered models only.

Requires running TurboCider HTTP and ComfyUI servers with this node pack.
Stores graph requests, ComfyUI history and generated images in --output-dir.
Never downloads or registers models. No automatic retry of submissions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings/python"))
from turbocider_http import HTTPClient


def exchange(url, path, data=None, binary=False, content_type="application/json"):
    if data is not None and not isinstance(data, bytes):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read()
    return body if binary else (json.loads(body) if body else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", help="Run only this named case; repeatable")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = HTTPClient(args.api_url).catalog()
    installed = {i["modelID"]: i for i in catalog["installations"]}
    info = exchange(args.comfy_url, "/object_info")
    for name in ("TurboCiderModel", "TurboCiderLoRA", "TurboCiderGenerate", "TurboCiderEdit", "TurboCiderRequest"):
        assert name in info, f"Missing node: {name}"
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if args.case and summary_path.is_file() else []
    summary = [item for item in summary if item["case"] not in (args.case or [])]
    reference = None
    reference2 = None
    cases = [
        ("flux-batch", "flux2-klein-4b", 4, None, "auto", 2, None, {}),
        ("flux-edit", "flux2-klein-4b", 4, None, "auto", 1, "image.edit", {}),
        ("flux-transform", "flux2-klein-4b", 4, None, "auto", 1, "image.transform", {}),
        ("z-base", "z-image-turbo", 8, None, "auto", 1, None, {}),
        ("z-lora-runtime", "z-image-turbo", 8, "distill_patch", "inference_time", 1, None, {"allow_approximation": True}),
        ("z-lora-merge", "z-image-turbo", 8, "distill_patch", "in_memory_merge", 1, None, {"allow_approximation": True}),
        ("z-base-after-lora", "z-image-turbo", 8, None, "auto", 1, None, {}),
        ("qwen-base", "qwen-image-2.1", 20, None, "auto", 1, None, {}),
        ("qwen-turbo", "qwen-image-2.1", 6, "viggle", "inference_time", 1, None, {"allow_approximation": True}),
        ("qwen-edit", "qwen-image-2.1", 6, "viggle", "inference_time", 1, "image.edit", {"allow_approximation": True, "qwen21_reference_size": 512}),
        ("qwen-multi-edit", "qwen-image-2.1", 6, "viggle", "inference_time", 1, "image.edit", {"allow_approximation": True, "qwen21_reference_size": 512}),
        ("qwen-ordinary-lora", "qwen-image-2.1", 20, "NSFW_Image_Edit", "inference_time", 1, None, {}),
        ("qwen-ordinary-edit", "qwen-image-2.1", 20, "NSFW_Image_Edit", "inference_time", 1, "image.edit", {"allow_approximation": True, "qwen21_reference_size": 512}),
        ("qwen-base-after-lora", "qwen-image-2.1", 20, None, "auto", 1, None, {}),
    ]
    if args.case and set(args.case) - {case[0] for case in cases}:
        parser.error("Unknown case name")
    def upload(data, name):
        boundary = uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
        result = exchange(args.comfy_url, "/upload/image", body, content_type="multipart/form-data; boundary=" + boundary)
        return result["name"]
    # Subset runs can use a prior saved reference, never a checkpoint.
    for name in ("flux-batch-0.png", "flux-batch-1.png"):
        path = args.output_dir / name
        if path.is_file():
            value = upload(path.read_bytes(), "tc-validation-" + name)
            if reference is None:
                reference = value
            else:
                reference2 = value
    for name, model, steps, lora, strategy, batch, operation, advanced in cases:
        if args.case and name not in args.case:
            continue
        if model not in installed:
            summary.append({"case": name, "skipped": "No registered model"})
            continue
        selected = installed[model]
        graph = {"1": {"class_type": "TurboCiderModel", "inputs": {"installation": selected["id"], "api_url": args.api_url, "token_env": "TURBOCIDER_API_TOKEN"}}}
        backend = ["1", 0]
        if lora:
            adapter = next((i for i in catalog["loras"] if i["modelID"] == model and lora in i["name"]), None)
            if adapter is None:
                summary.append({"case": name, "skipped": "No matching registered LoRA"})
                continue
            graph["2"] = {"class_type": "TurboCiderLoRA", "inputs": {"backend": backend, "lora": adapter["id"], "strength": 0.6 if lora == "NSFW_Image_Edit" else 1.0, "strategy": strategy}}
            backend = ["2", 0]
        inputs = {"backend": backend, "prompt": "A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.",
                  "width": 512, "height": 512, "steps": steps, "seed": 42, "batch_size": batch, "advanced": json.dumps(advanced), "timeout_seconds": 1800}
        if operation:
            if reference is None:
                raise RuntimeError("Run flux-batch first to create reference images")
            graph["5"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
            inputs.update(images=["5", 0], operation=operation, strength=0.65,
                          prompt="Change the blue teapot to a red ceramic teapot, preserve the table and lighting.")
            if name == "qwen-multi-edit":
                graph["6"] = {"class_type": "LoadImage", "inputs": {"image": reference2 or reference}}
                graph["7"] = {"class_type": "ImageBatch", "inputs": {"image1": ["5", 0], "image2": ["6", 0]}}
                inputs.update(images=["7", 0], prompt="Place the teapot from image 1 beside the teapot from image 2 on one wooden table.")
        graph["3"] = {"class_type": "TurboCiderEdit" if operation else "TurboCiderGenerate", "inputs": inputs}
        graph["4"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "TurboCider-validation/" + name}}
        (args.output_dir / (name + "-prompt.json")).write_text(json.dumps(graph, indent=2))
        start = time.monotonic()
        response = exchange(args.comfy_url, "/prompt", {"prompt": graph, "client_id": "turbocider-validation"})
        prompt_id = response["prompt_id"]
        print(json.dumps({"case": name, "prompt_id": prompt_id, "state": "submitted"}), flush=True)
        deadline = time.monotonic() + 2400
        while time.monotonic() < deadline:
            history = exchange(args.comfy_url, "/history/" + prompt_id)
            if prompt_id in history:
                record = history[prompt_id]
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out; inspect queue")
        (args.output_dir / (name + "-history.json")).write_text(json.dumps(record, indent=2))
        result = {"case": name, "prompt_id": prompt_id, "seconds": round(time.monotonic() - start, 3), "status": record["status"]["status_str"], "images": []}
        for index, image in enumerate(record.get("outputs", {}).get("4", {}).get("images", [])):
            from urllib.parse import urlencode
            data = exchange(args.comfy_url, "/view?" + urlencode(image), binary=True)
            path = args.output_dir / f"{name}-{index}.png"
            path.write_bytes(data)
            result["images"].append({"file": path.name, "sha256": hashlib.sha256(data).hexdigest()})
            if name == "flux-batch":
                value = upload(data, "tc-validation-" + path.name)
                if index == 0:
                    reference = value
                else:
                    reference2 = value
        summary.append(result)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(result), flush=True)
        if result["status"] != "success" or len(result["images"]) != batch:
            raise RuntimeError(f"{name} failed; see history")
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Validation finished. No weights downloaded.", flush=True)


if __name__ == "__main__":
    main()
