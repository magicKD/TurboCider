"""A diagnostic-only ComfyUI node, copied into an owned isolated instance.

The sampler remains ComfyUI's stock Euler implementation. Its existing
callback is observed to persist the pre-step state and denoised output. This
node must not be loaded into a shared ComfyUI process or used for timings.
"""
from pathlib import Path
import re

import comfy.samplers
import folder_paths
import torch
from safetensors.torch import save_file


class LTX098EulerTrace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trace_name": ("STRING", {"default": "trace-0"})},
                "optional": {"capture_decode_noise": ("BOOLEAN", {"default": False}),
                             "capture_first_ffn": ("BOOLEAN", {"default": False}),
                             "capture_ffn_blocks": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "diagnostics"

    def get_sampler(self, trace_name, capture_decode_noise=False, capture_first_ffn=False, capture_ffn_blocks=""):
        if not re.fullmatch(r"trace-[0-9]+", trace_name):
            raise ValueError("invalid trace name")
        root = Path(folder_paths.get_output_directory()) / "diagnostic" / trace_name
        root.mkdir(parents=True, exist_ok=False)
        if capture_first_ffn and capture_ffn_blocks:
            raise ValueError("choose one FFN capture mode")
        if capture_first_ffn or capture_ffn_blocks:
            # Capture selected blocks in the first denoising step, then
            # restore the method; the isolated server has no concurrent jobs.
            from comfy.ldm.lightricks.model import FeedForward
            original_forward = FeedForward.forward
            selected = {0} if capture_first_ffn else {int(value) for value in capture_ffn_blocks.split(",")}
            if not selected or any(index < 0 or index >= 28 for index in selected):
                raise ValueError("FFN capture blocks must be in [0,28)")
            calls = 0

            def traced_forward(module, value):
                nonlocal calls
                index = calls
                calls += 1
                if index in selected:
                    filename = "first_ffn_input.safetensors" if capture_first_ffn else f"ffn_{index}_input.safetensors"
                    save_file({"input": value.detach().cpu().contiguous()}, str(root / filename))
                if index == max(selected):
                    FeedForward.forward = original_forward
                return original_forward(module, value)

            FeedForward.forward = traced_forward
        if capture_decode_noise:
            # This owned one-request server has no concurrent workflows. Observe
            # the existing random draw without changing its value or RNG state.
            from comfy.ldm.lightricks.vae.causal_video_autoencoder import VideoVAE
            original_decode = VideoVAE.decode

            def traced_decode(vae, *args, **kwargs):
                VideoVAE.decode = original_decode
                original_random = torch.randn_like
                noise = []

                def traced_random(*random_args, **random_kwargs):
                    value = original_random(*random_args, **random_kwargs)
                    noise.append(value.detach().cpu().contiguous())
                    return value

                torch.randn_like = traced_random
                try:
                    result = original_decode(vae, *args, **kwargs)
                finally:
                    torch.randn_like = original_random
                if len(noise) != 1:
                    raise ValueError(f"expected one decoder noise draw, got {len(noise)}")
                save_file({"noise": noise[0]}, str(root / "decode_noise.safetensors"))
                return result

            VideoVAE.decode = traced_decode
        sampler = comfy.samplers.sampler_object("euler")
        original = sampler.sampler_function

        def traced_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
            def trace_step(data):
                index = data["i"]
                values = {name: data[name].detach().cpu().contiguous()
                          for name in ("x", "denoised", "sigma", "sigma_hat")}
                path = root / f"step_{index}.safetensors"
                if path.exists():
                    raise FileExistsError(path)
                save_file(values, str(path))
                if callback is not None:
                    callback(data)

            return original(model, x, sigmas, extra_args=extra_args, callback=trace_step,
                            disable=disable, **options)

        sampler.sampler_function = traced_euler
        return (sampler,)


NODE_CLASS_MAPPINGS = {"LTX098EulerTrace": LTX098EulerTrace}
