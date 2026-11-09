"""Diagnostic-only ComfyUI sampler for FastWan2.2's three-step DMD path.

The model, text encoder, first-frame VAE and decoder remain stock ComfyUI.
This local node supplies the explicit DMD re-noising between model calls;
stock Euler with three steps is a different sampler.
"""
import torch
import comfy.samplers


def sample_dmd(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
    del disable, options
    extra_args = {} if extra_args is None else extra_args
    if len(sigmas) != 4 or float(sigmas[-1]) != 0:
        raise ValueError("FastWan2.2 requires three positive sigmas followed by zero")
    values = [float(value) for value in sigmas]
    if not (1.0 >= values[0] > values[1] > values[2] > 0):
        raise ValueError("FastWan2.2 sigmas must decrease from at most one")
    seed = extra_args.get("seed")
    if seed is None:
        raise ValueError("FastWan2.2 DMD sampler needs the explicit noise seed")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # RandomNoise made the first draw from the same seeded CPU generator.
    torch.randn(tuple(x.shape), generator=generator, dtype=torch.float32, device="cpu")
    batch = x.new_ones([x.shape[0]])
    for index in range(3):
        sigma = sigmas[index]
        denoised = model(x, sigma * batch, **extra_args)
        if callback is not None:
            callback({"x": x, "i": index, "sigma": sigma,
                      "sigma_hat": sigma, "denoised": denoised})
        if index == 2:
            x = denoised
        else:
            next_sigma = sigmas[index + 1]
            fresh = torch.randn(tuple(x.shape), generator=generator,
                                dtype=torch.float32, device="cpu").to(device=x.device, dtype=x.dtype)
            x = (1.0 - next_sigma) * denoised + next_sigma * fresh
    return x


def sample_quanhaol_dmd(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
    """Diagnostic four-step path; CPU noise is shared with the native probe."""
    del disable, options
    extra_args = {} if extra_args is None else extra_args
    values = [float(value) for value in sigmas]
    if len(values) != 5 or values[-1] != 0 or not all(
            abs(a - b) < 1e-6 for a, b in zip(values, (1., .96, 8 / 9, 8 / 11, 0.))):
        raise ValueError("quanhaol Wan2.2 requires the four-step shift-8 schedule")
    seed = extra_args.get("seed")
    if seed is None:
        raise ValueError("quanhaol Wan2.2 sampler needs the explicit noise seed")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    torch.randn(tuple(x.shape), generator=generator, dtype=torch.float32, device="cpu")
    x = x.to(torch.bfloat16)
    batch = x.new_ones([x.shape[0]], dtype=torch.float32)
    for index, timestep in enumerate((1000., 960., 888., 728.)):
        sigma = sigmas[index]
        model_sigma = timestep / 1000.
        x_float = x.float()
        model_denoised = model(x_float, model_sigma * batch, **extra_args)
        # Comfy returns x0 at the model timestep. Undo that conversion to use
        # the unrounded scheduler sigma, then restore its inpaint condition.
        denoised = x_float - (x_float - model_denoised.float()) * (sigma / model_sigma)
        mask = extra_args.get("denoise_mask")
        if mask is not None:
            denoised = torch.where(mask > 0, denoised, model_denoised.float())
        denoised = denoised.to(torch.bfloat16)
        if callback is not None:
            callback({"x": x, "i": index, "sigma": sigma,
                      "sigma_hat": sigma, "denoised": denoised})
        if index == 3:
            x = denoised
        else:
            fresh = torch.randn(tuple(x.shape), generator=generator, dtype=torch.float32, device="cpu")
            fresh = fresh.to(device=x.device, dtype=torch.bfloat16)
            next_sigma = sigmas[index + 1].to(dtype=torch.bfloat16)
            x = (1.0 - next_sigma) * denoised + next_sigma * fresh
    return x


class QuanhaolWan22DMDSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "sampler"
    CATEGORY = "diagnostics"

    def sampler(self):
        import json
        from pathlib import Path
        import folder_paths

        sampler = comfy.samplers.sampler_object("euler")
        def recorded(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
            steps = []
            def observed(data):
                steps.append({"index": int(data["i"]), "sigma": float(data["sigma"])})
                if callback is not None:
                    callback(data)
            result = sample_quanhaol_dmd(model, x, sigmas, extra_args, observed, disable, **options)
            directory = Path(folder_paths.get_output_directory()) / "diagnostic"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "fastwan22-dmd-steps.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps({"steps": steps, "seed": int(extra_args["seed"])}, indent=2) + "\n")
            return result
        sampler.sampler_function = recorded
        return (sampler,)


class FastWan22DMDSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"capture_first_block": ("BOOLEAN", {"default": False}),
                                             "capture_transformer": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "sampler"
    CATEGORY = "diagnostics"

    def sampler(self, capture_first_block=False, capture_transformer=False):
        import json
        from pathlib import Path
        import folder_paths

        sampler = comfy.samplers.sampler_object("euler")
        def recorded(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
            steps = []
            def observed(data):
                steps.append({"index": int(data["i"]), "sigma": float(data["sigma"])})
                if callback is not None:
                    callback(data)
            if capture_transformer:
                from comfy.ldm.wan.model import WanModel
                original_transformer = WanModel.forward
                transformer_captured = False

                def capture_model(instance, latents, timestep, context, *args, **kwargs):
                    nonlocal transformer_captured
                    if transformer_captured:
                        return original_transformer(instance, latents, timestep, context, *args, **kwargs)
                    transformer_captured = True
                    from safetensors.torch import save_file
                    from comfy.ldm.wan.model import WanAttentionBlock

                    block_forward = WanAttentionBlock.forward
                    trace = {}
                    detail = {}

                    def traced_block(block, hidden, e, freqs, context, *block_args, **block_kwargs):
                        index = block_kwargs.get("transformer_options", {}).get("block_index")
                        hooks = []
                        if index == 0:
                            if e.ndim != 4 or e.shape[1] < 2 or \
                                    not torch.equal(e[:, 1:], e[:, 1:2].expand_as(e[:, 1:])):
                                raise ValueError("unexpected transformer trace timestep groups")
                            trace.update(hidden_input=hidden.detach().cpu().contiguous(),
                                         context=context.detach().cpu().contiguous(),
                                         modulation=torch.cat((e[:, :1], e[:, 1:2]), 1).detach().cpu().contiguous(),
                                         cosine=freqs[0, :, 0, :, 0, 0].repeat_interleave(2, dim=-1).detach().cpu().contiguous(),
                                         sine=freqs[0, :, 0, :, 1, 0].repeat_interleave(2, dim=-1).detach().cpu().contiguous())
                            def record(name, input_name=None):
                                def hook(_module, inputs, output):
                                    if input_name:
                                        detail[input_name] = inputs[0].detach().cpu().contiguous()
                                    detail[name] = output.detach().cpu().contiguous()
                                return hook
                            for label, attention in (("self", block.self_attn), ("cross", block.cross_attn)):
                                hooks.extend((attention.q.register_forward_hook(record(label + "_q", label + "_input")),
                                              attention.k.register_forward_hook(record(label + "_k")),
                                              attention.v.register_forward_hook(record(label + "_v")),
                                              attention.norm_q.register_forward_hook(record(label + "_q_norm")),
                                              attention.norm_k.register_forward_hook(record(label + "_k_norm")),
                                              attention.o.register_forward_hook(record(label + "_output", label + "_attended"))))
                            hooks.extend((block.norm3.register_forward_hook(record("cross_norm", "self_residual")),
                                          block.norm2.register_forward_hook(record("ffn_norm", "cross_residual")),
                                          block.ffn[0].register_forward_hook(record("ffn_up", "ffn_input")),
                                          block.ffn[2].register_forward_hook(record("ffn_output", "ffn_activated"))))
                        try:
                            value = block_forward(block, hidden, e, freqs, context, *block_args, **block_kwargs)
                        finally:
                            for hook in hooks:
                                hook.remove()
                        if index is not None:
                            trace[f"block_{index}"] = value.detach().cpu().contiguous()
                        return value

                    WanAttentionBlock.forward = traced_block
                    try:
                        result = original_transformer(instance, latents, timestep, context, *args, **kwargs)
                    finally:
                        WanAttentionBlock.forward = block_forward

                    if timestep.ndim != 2 or timestep.shape[0] != 1 or timestep.shape[1] < 2 or \
                            timestep[0, 0].item() != 0 or \
                            not torch.equal(timestep[:, 1:], timestep[:, 1:2].expand_as(timestep[:, 1:])):
                        raise ValueError(f"unexpected first-frame timestep pattern: {timestep}")
                    directory = Path(folder_paths.get_output_directory()) / "diagnostic"
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "fastwan22-first-transformer.safetensors"
                    if path.exists():
                        raise FileExistsError(path)
                    save_file({"latents": latents.detach().cpu().contiguous(),
                               "text": context.detach().cpu().contiguous(),
                               "expected": result.detach().cpu().contiguous()}, str(path),
                              metadata={"timestep": str(float(timestep[0, 1]))})
                    if len(trace) != 35:
                        raise ValueError(f"incomplete transformer trace: {sorted(trace)}")
                    save_file(trace, str(directory / "fastwan22-transformer-trace.safetensors"))
                    save_file(detail, str(directory / "fastwan22-block0-detail.safetensors"))
                    return result

                WanModel.forward = capture_model
            if capture_first_block:
                # This node is installed only in an owned, isolated ComfyUI
                # process.  Restore the method even if sampling fails.
                from comfy.ldm.wan.model import WanAttentionBlock
                original = WanAttentionBlock.forward
                captured = False

                def capture(block, hidden, e, freqs, context, *args, **kwargs):
                    nonlocal captured
                    result = original(block, hidden, e, freqs, context, *args, **kwargs)
                    if captured:
                        return result
                    captured = True
                    from safetensors.torch import save_file

                    grid = kwargs.get("transformer_options", {}).get("grid_sizes")
                    if grid is None or len(grid) != 3 or e.ndim != 4 or e.shape[1] != grid[0]:
                        raise ValueError(f"unexpected Wan2.2 block geometry: grid={grid}, e={e.shape}")
                    if e.shape[1] < 2 or not torch.equal(e[:, 1:], e[:, 1:2].expand_as(e[:, 1:])):
                        raise ValueError("motion-frame timestep modulation is not uniform")
                    if freqs.ndim != 6 or freqs.shape[0] != 1 or freqs.shape[2] != 1 or freqs.shape[-2:] != (2, 2):
                        raise ValueError(f"unexpected Wan2.2 rotary geometry: {freqs.shape}")
                    cosine = freqs[0, :, 0, :, 0, 0].repeat_interleave(2, dim=-1)
                    sine = freqs[0, :, 0, :, 1, 0].repeat_interleave(2, dim=-1)
                    modulation = torch.cat((e[:, :1], e[:, 1:2]), dim=1)
                    tensors = {"hidden": hidden, "context": context, "modulation": modulation,
                               "cosine": cosine, "sine": sine, "expected": result}
                    directory = Path(folder_paths.get_output_directory()) / "diagnostic"
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "fastwan22-first-block.safetensors"
                    if path.exists():
                        raise FileExistsError(path)
                    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(path),
                              metadata={"first_frame_tokens": str(grid[1] * grid[2]),
                                        "grid": ",".join(map(str, grid))})
                    return result

                WanAttentionBlock.forward = capture
            try:
                result = sample_dmd(model, x, sigmas, extra_args=extra_args,
                                    callback=observed, disable=disable, **options)
                if capture_first_block and not captured:
                    raise RuntimeError("Wan2.2 first block was not captured")
                if capture_transformer and not transformer_captured:
                    raise RuntimeError("Wan2.2 transformer was not captured")
            finally:
                if capture_first_block:
                    WanAttentionBlock.forward = original
                if capture_transformer:
                    WanModel.forward = original_transformer
            path = Path(folder_paths.get_output_directory()) / "diagnostic" / "fastwan22-dmd-steps.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as output:
                json.dump({"steps": steps, "seed": int(extra_args["seed"])}, output, indent=2)
                output.write("\n")
            return result
        sampler.sampler_function = recorded
        return (sampler,)


class FastWan22CaptureVAE:
    """Capture the exact single-frame VAE input/output in an owned server."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "capture"
    CATEGORY = "diagnostics"

    def capture(self, vae):
        from pathlib import Path
        from safetensors.torch import save_file
        import folder_paths
        from comfy.ldm.wan.vae2_2 import WanVAE

        stage = vae.first_stage_model
        if not isinstance(stage, WanVAE):
            raise ValueError("VAE capture requires the stock Wan2.2 VAE")
        original = stage.encode
        original_decode = stage.decode

        def observed(pixels, *args, **kwargs):
            try:
                result = original(pixels, *args, **kwargs)
                if pixels.ndim != 5 or tuple(pixels.shape[:3]) != (1, 3, 1) or result.ndim != 5:
                    raise ValueError(f"unexpected single-frame VAE geometry: {pixels.shape} -> {result.shape}")
                directory = Path(folder_paths.get_output_directory()) / "diagnostic"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "fastwan22-first-frame-vae.safetensors"
                if path.exists():
                    raise FileExistsError(path)
                save_file({"pixels": pixels.detach().cpu().contiguous(),
                           "expected": result.detach().cpu().contiguous()}, str(path))
                return result
            finally:
                stage.encode = original

        stage.encode = observed

        def observed_decode(latents, *args, **kwargs):
            try:
                result = original_decode(latents, *args, **kwargs)
                if latents.ndim != 5 or latents.shape[:2] != (1, 48) or result.ndim != 5:
                    raise ValueError(f"unexpected Wan2.2 VAE decode geometry: {latents.shape} -> {result.shape}")
                directory = Path(folder_paths.get_output_directory()) / "diagnostic"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "fastwan22-video-vae-decode.safetensors"
                if path.exists():
                    raise FileExistsError(path)
                save_file({"latents": latents.detach().cpu().contiguous(),
                           "expected": result.detach().cpu().contiguous()}, str(path))
                return result
            finally:
                stage.decode = original_decode

        stage.decode = observed_decode
        return (vae,)


class FastWan22CaptureTextEncode:
    """Capture stock ComfyUI UMT5 tokens and conditioning for parity."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "diagnostics"

    def encode(self, clip, text):
        from pathlib import Path
        from safetensors.torch import save_file
        import folder_paths

        tokens = clip.tokenize(text)
        batches = tokens.get("umt5xxl")
        if not isinstance(batches, list) or len(batches) != 1 or not 512 <= len(batches[0]) <= 4096:
            raise ValueError("text capture requires one bounded UMT5 batch of at least 512 tokens")
        if any(not isinstance(entry[0], int) or entry[1] != 1.0 for entry in batches[0]):
            raise ValueError("text capture requires unweighted token IDs")
        ids = [entry[0] for entry in batches[0]]
        valid = ids.index(1) + 1
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(conditioning) != 1:
            raise ValueError("text capture requires one conditioning result")
        directory = Path(folder_paths.get_output_directory()) / "diagnostic"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "fastwan22-text.safetensors"
        if path.exists():
            raise FileExistsError(path)
        save_file({"ids": torch.tensor(ids, dtype=torch.int32),
                   "valid": torch.tensor(valid, dtype=torch.int32),
                   "expected": conditioning[0][0].detach().cpu().contiguous()}, str(path))
        return (conditioning,)


NODE_CLASS_MAPPINGS = {"FastWan22DMDSampler": FastWan22DMDSampler,
                       "QuanhaolWan22DMDSampler": QuanhaolWan22DMDSampler,
                       "FastWan22CaptureVAE": FastWan22CaptureVAE,
                       "FastWan22CaptureTextEncode": FastWan22CaptureTextEncode}
