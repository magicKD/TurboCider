from __future__ import annotations
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

# Installed packs receive the one canonical HTTP client from the installer.
try:
    from .turbocider_http import HTTPClient
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bindings/python"))
    from turbocider_http import HTTPClient


def default_client():
    return HTTPClient(os.environ.get("TURBOCIDER_API_URL", "http://127.0.0.1:8765"), os.environ.get("TURBOCIDER_API_TOKEN", ""), timeout=15)


def choices(kind):
    try:
        return [f'{i["id"]} | {i["modelID"]} | {i["name"]}' for i in default_client().catalog()[kind]] or ["unavailable"]
    except Exception:
        # ComfyUI can still start while TurboCider is offline. Execution reports
        # the connection error; Refresh reloads catalog after the API starts.
        return ["unavailable"]


def identity(selection):
    return selection.split(" | ", 1)[0]


def strict_object(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate option: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    result = json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("Advanced options must be a JSON object")
    return result


def client_for(backend):
    return HTTPClient(backend["url"], backend["token"])


def check_interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def wait_job(client, job_id, timeout):
    from comfy.utils import ProgressBar
    progress = ProgressBar(1)
    deadline = time.monotonic() + timeout
    last_phase = None
    try:
        while time.monotonic() < deadline:
            check_interrupt()
            job = client.status(job_id)
            if job["state"] == "succeeded":
                return job
            if job["state"] in ("failed", "cancelled", "interrupted"):
                raise RuntimeError(f'TurboCider job {job_id} {job["state"]}: {job.get("error", "")}')
            event = job.get("progress", {})
            completed, total = event.get("completed", 0), event.get("total", 0)
            if isinstance(total, (int, float)) and total > 0 and isinstance(completed, (int, float)):
                progress.update_absolute(int(completed), int(total))
            phase = event.get("phase")
            if phase and phase != last_phase:
                print(f"[TurboCider] {job_id}: {phase}", flush=True)
                last_phase = phase
            time.sleep(0.25)
        raise TimeoutError(f"TurboCider job {job_id} exceeded {timeout}s")
    except BaseException as error:
        # Keep assets and receipt for safe-boundary cancellation and diagnosis.
        # Never remove an input while a native job may still be reading it.
        try:
            client.cancel(job_id)
        except Exception as cancel_error:
            print(f"[TurboCider] Could not cancel {job_id}: {cancel_error}; inspect API history", flush=True)
        if hasattr(error, "add_note"):
            error.add_note(f"TurboCider job ID: {job_id}; cancellation requested; inspect API status for terminal state")
        raise


def upload_images(client, images, role="reference"):
    import numpy as np
    from PIL import Image
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("Expected ComfyUI IMAGE [batch, height, width, RGB/RGBA]")
    inputs = []
    for image in images:
        check_interrupt()
        values = image.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Input image has non-finite pixels")
        buffer = io.BytesIO()
        Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(buffer, format="PNG")
        inputs.append({"kind": "image", "role": role, "asset_id": client.upload(buffer.getvalue())})
    return inputs


def decode_image(data):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        values = np.asarray(image).astype(np.float32) / 255.0
    # Comfy masks are inverse alpha; preserve transparency as a separate output.
    return torch.from_numpy(values[..., :3].copy()).unsqueeze(0), torch.from_numpy(1 - values[..., 3].copy()).unsqueeze(0)


class TurboCiderModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "installation": (choices("installations"),),
            "api_url": ("STRING", {"default": os.environ.get("TURBOCIDER_API_URL", "http://127.0.0.1:8765")}),
            "token_env": ("STRING", {"default": "TURBOCIDER_API_TOKEN"}),
        }, "optional": {"installation_id": ("STRING", {"default": "", "tooltip": "Optional explicit registration ID for another API endpoint"})}}
    RETURN_TYPES = ("TURBOCIDER_MODEL", "STRING")
    RETURN_NAMES = ("backend", "capabilities")
    FUNCTION = "connect"
    CATEGORY = "TurboCider"

    @classmethod
    def VALIDATE_INPUTS(cls, installation):
        # Saved workflows remain usable when the API was offline at UI startup;
        # the actual backend catalog is authoritative during execution.
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Refresh inventory/defaults and token each execution.

    def connect(self, installation, api_url, token_env="TURBOCIDER_API_TOKEN", installation_id=""):
        client = HTTPClient(api_url, os.environ.get(token_env, ""))
        catalog = client.catalog()
        key = installation_id.strip() or identity(installation)
        selected = next((i for i in catalog["installations"] if i["id"] == key), None)
        if selected is None:
            raise ValueError("Select an installed TurboCider model and refresh ComfyUI node definitions")
        model = next(m for m in catalog["models"] if m["id"] == selected["modelID"])
        return ({"url": api_url, "token": client.token, "installation_id": key, "model": model, "loras": [], "catalog_loras": catalog["loras"]}, json.dumps(model, ensure_ascii=False, indent=2))


class TurboCiderLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"backend": ("TURBOCIDER_MODEL",), "lora": (choices("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.05}),
            "strategy": (["auto", "inference_time", "in_memory_merge", "disk_premerge"],)},
            "optional": {"lora_id": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("TURBOCIDER_MODEL",)
    FUNCTION = "attach"
    CATEGORY = "TurboCider"

    @classmethod
    def VALIDATE_INPUTS(cls, lora):
        return True

    def attach(self, backend, lora, strength, strategy="auto", lora_id=""):
        if not math.isfinite(strength):
            raise ValueError("LoRA strength must be finite")
        model = backend["model"]
        if not model.get("supports_lora"):
            raise ValueError(f'{model["id"]} does not support LoRA')
        selected = next((i for i in backend["catalog_loras"] if i["id"] == (lora_id.strip() or identity(lora))), None)
        if selected is None or selected["modelID"] != model["id"]:
            raise ValueError("LoRA registration does not belong to the selected model")
        resolved = model.get("default_lora_strategy") if strategy == "auto" else strategy
        if resolved not in model.get("lora_strategies", []):
            raise ValueError(f'{model["id"]} supports LoRA strategies: {model.get("lora_strategies", [])}')
        if backend.get("loras") and backend["lora_strategy"] != resolved:
            raise ValueError("All LoRAs in a chain must use the same strategy")
        return ({**backend, "loras": [*backend["loras"], {"id": selected["id"], "role": "transformer", "strength": strength}], "lora_strategy": resolved},)


def compose(backend, prompt, width, height, steps, seed, advanced, operation, inputs=None):
    model = backend["model"]
    if operation not in model.get("executor_operations", model["operations"]):
        raise ValueError(f'{model["id"]} does not support {operation}')
    options = strict_object(advanced)
    reserved = {"schema_version", "model", "model_path", "output", "outputs", "inputs", "loras", "operation", "prompt", "width", "height", "steps", "seed"}
    if reserved.intersection(options):
        raise ValueError("Advanced options cannot replace node-owned fields: " + ", ".join(sorted(reserved.intersection(options))))
    request = {"schema_version": 1, "model": model["id"], "operation": operation, "prompt": prompt,
               "width": width, "height": height, "steps": steps or model["default_steps"], "seed": seed,
               "execution": "gpu", "residency": model["default_residency"], "prompt_enhance": False,
               **options}
    if inputs:
        request["inputs"] = inputs
    if backend["loras"]:
        request["loras"] = backend["loras"]
        request["lora_strategy"] = backend["lora_strategy"]
    return request


class TurboCiderGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"backend": ("TURBOCIDER_MODEL",),
            "prompt": ("STRING", {"multiline": True, "default": "A red fox in a snowy forest, soft morning light."}),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
            "steps": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "0 uses backend model default; Qwen Viggle requires 6"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "control_after_generate": True}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "advanced": ("STRING", {"multiline": True, "default": "{}"}),
            "timeout_seconds": ("INT", {"default": 1800, "min": 1, "max": 86400})}}
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "alpha_mask", "jobs")
    FUNCTION = "generate"
    CATEGORY = "TurboCider"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Explicit queued generations are never suppressed by cache.

    def generate(self, backend, prompt, width, height, steps, seed, batch_size=1, advanced="{}", timeout_seconds=1800, images=None, operation="image.generate", strength=1.0):
        import torch
        client = client_for(backend)
        # Reject unsupported editing before uploading inputs.
        request = compose(backend, prompt, width, height, steps, seed, advanced, operation)
        if images is not None:
            role = "init_image" if operation == "image.transform" else "reference"
            request["inputs"] = upload_images(client, images, role)
        if operation == "image.transform":
            for item in request.get("inputs", []):
                item["strength"] = strength
        client.plan(backend["installation_id"], request)
        results, masks, receipts = [], [], []
        for index in range(batch_size):
            check_interrupt()
            request["seed"] = (seed + index) % (2 ** 31)
            job_id = client.submit(backend["installation_id"], request)
            print(f"[TurboCider] submitted {job_id}", flush=True)
            job = wait_job(client, job_id, timeout_seconds)
            image, mask = decode_image(client.artifact(job_id))
            results.append(image)
            masks.append(mask)
            receipts.append(job)
        return torch.cat(results), torch.cat(masks), json.dumps(receipts, ensure_ascii=False)


class TurboCiderEdit(TurboCiderGenerate):
    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        spec["required"].update({"images": ("IMAGE",), "operation": (["image.edit", "image.transform"],),
                                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01})})
        return spec


class TurboCiderRequest:
    """Advanced all-executor API access, including video. Output file is local to ComfyUI."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"backend": ("TURBOCIDER_MODEL",),
            "request_json": ("STRING", {"multiline": True, "default": '{"operation":"video.generate","prompt":"A fox walking in snow","audio":false}'}),
            "timeout_seconds": ("INT", {"default": 3600, "min": 1, "max": 86400})},
            "optional": {"images": ("IMAGE",), "image_role": (["reference", "init_image", "first_frame", "last_frame"],)}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("artifact_path", "job")
    FUNCTION = "run"
    CATEGORY = "TurboCider/Advanced"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, backend, request_json, timeout_seconds=3600, images=None, image_role="reference"):
        import folder_paths
        client = client_for(backend)
        request = strict_object(request_json)
        if backend["loras"]:
            if "loras" in request or "lora_strategy" in request:
                raise ValueError("Choose LoRA nodes or request JSON LoRAs, not both")
            request.update(loras=backend["loras"], lora_strategy=backend["lora_strategy"])
        if images is not None:
            if "inputs" in request:
                raise ValueError("Choose IMAGE input or JSON inputs, not both")
            request["inputs"] = upload_images(client, images, image_role)
        client.plan(backend["installation_id"], request)
        check_interrupt()
        job_id = client.submit(backend["installation_id"], request)
        job = wait_job(client, job_id, timeout_seconds)
        suffix = ".mp4" if backend["model"]["output"] == "video" else ".png"
        output = Path(folder_paths.get_output_directory()) / ("TurboCider_" + uuid.uuid4().hex + suffix)
        output.write_bytes(client.artifact(job_id))
        return str(output), json.dumps(job, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {"TurboCiderModel": TurboCiderModel, "TurboCiderLoRA": TurboCiderLoRA,
                       "TurboCiderGenerate": TurboCiderGenerate, "TurboCiderEdit": TurboCiderEdit,
                       "TurboCiderRequest": TurboCiderRequest}
NODE_DISPLAY_NAME_MAPPINGS = {"TurboCiderModel": "TurboCider · Model (API)", "TurboCiderLoRA": "TurboCider · LoRA (API)",
    "TurboCiderGenerate": "TurboCider · Generate Image", "TurboCiderEdit": "TurboCider · Edit / Transform Image",
    "TurboCiderRequest": "TurboCider · Native Request / Video"}
