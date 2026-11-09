"""Diagnostic-only ComfyUI node to match the native second random draw."""
import torch
from pathlib import Path
import comfy.samplers
import folder_paths
from safetensors.torch import save_file


class _SequentialNoise:
    def __init__(self, seed, first_height, first_width):
        self.seed = seed
        self.first_height = first_height
        self.first_width = first_width

    def generate_noise(self, latent):
        samples = latent["samples"]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        torch.randn((samples.shape[0], samples.shape[1], samples.shape[2], self.first_height, self.first_width),
                    generator=generator, device="cpu", dtype=torch.float32)
        return torch.randn(samples.shape, generator=generator, device="cpu", dtype=torch.float32)


class LTX098AdaIN:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"upsampled": ("LATENT",), "reference": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "diagnostics"

    def apply(self, upsampled, reference):
        value = upsampled["samples"].to(torch.bfloat16)
        original = reference["samples"].to(torch.bfloat16)
        # Match Lightricks adain_filter_latent: per-channel sample std/mean
        # over (frame,height,width), after the learned spatial upsampler.
        source_std, source_mean = torch.std_mean(original, dim=(2, 3, 4), keepdim=True)
        target_std, target_mean = torch.std_mean(value, dim=(2, 3, 4), keepdim=True)
        adjusted = ((value - target_mean) / target_std.clamp_min(1e-6)) * source_std + source_mean
        return ({**upsampled, "samples": adjusted.float()},)


class LTX098SecondNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                             "first_height": ("INT", {"default": 10, "min": 1}),
                             "first_width": ("INT", {"default": 10, "min": 1})}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "noise"
    CATEGORY = "diagnostics"

    def noise(self, seed, first_height, first_width):
        return (_SequentialNoise(seed, first_height, first_width),)


class LTX098TraceSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stage": (["first", "second"],)}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "sampler"
    CATEGORY = "diagnostics"

    def sampler(self, stage):
        root = Path(folder_paths.get_output_directory()) / "diagnostic" / (stage + "_steps")
        root.mkdir(parents=True, exist_ok=False)
        sampler = comfy.samplers.sampler_object("euler")
        original = sampler.sampler_function

        def trace(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
            def step(data):
                save_file({name: data[name].detach().cpu().contiguous() for name in ("x", "denoised")},
                          str(root / f"step_{data['i']}.safetensors"))
                if callback is not None:
                    callback(data)
            return original(model, x, sigmas, extra_args=extra_args, callback=step, disable=disable, **options)
        sampler.sampler_function = trace
        return (sampler,)


NODE_CLASS_MAPPINGS = {"LTX098SecondNoise": LTX098SecondNoise, "LTX098AdaIN": LTX098AdaIN,
                       "LTX098TraceSampler": LTX098TraceSampler}
