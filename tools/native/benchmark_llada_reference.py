#!/usr/bin/env python3
"""Run the official LLaDA-Image-Turbo Diffusers reference locally.

This is deliberately a reference/acceptance tool, not the native executor.  It
keeps the upstream pipeline as the oracle while TurboCider's native runtime is
implemented and benchmarked separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference-root", type=Path,
                        default=root.parent / "references" / "LLaDA-Image")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-mode", choices=("text", "editing"), default="text")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--keep-warmup", action="store_true",
                        help="keep warmup PNGs next to the requested output for parity checks")
    parser.add_argument("--dump-dir", type=Path,
                        help="write reference conditioning/noise/step latents as safetensors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.generation_mode == "editing" and args.input_image is None:
        raise SystemExit("--input-image is required for --generation-mode editing")
    if args.generation_mode == "text" and args.input_image is not None:
        raise SystemExit("--input-image is only valid for editing")
    if args.width % 16 or args.height % 16:
        raise SystemExit("text dimensions must be multiples of 16")
    if args.generation_mode == "editing" and (args.width % 32 or args.height % 32):
        raise SystemExit("editing dimensions must be multiples of 32")

    reference_root = args.reference_root.resolve()
    sys.path.insert(0, str(reference_root))
    import torch
    from PIL import Image
    from src import LLaDAImagePipeline

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    load_start = time.perf_counter()
    pipeline = LLaDAImagePipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, device=args.device
    )
    load_seconds = time.perf_counter() - load_start

    def generate(output: Path, seed: int, dump_dir: Path | None = None) -> float:
        latent_shape = (
            1,
            pipeline.transformer.config.in_channels,
            args.height // pipeline.latent_scale_factor,
            args.width // pipeline.latent_scale_factor,
        )
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(latent_shape, generator=cpu_generator, dtype=torch.float32)
        latents = latents.to(pipeline.transformer.device).to(pipeline.transformer.dtype).float()
        with torch.inference_mode():
            conditioning = pipeline.encode_prompt(
                prompt=args.prompt,
                negative_prompt=None,
                do_classifier_free_guidance=args.guidance_scale > 1.0,
                num_images_per_prompt=1,
                max_sequence_length=2048,
                device=pipeline.transformer.device,
            )
        captured_model_outputs = []
        captured_latents = []
        captured_transformer_stages = {}
        captured_text_stages = {}
        stage_hooks = []
        if dump_dir:
            from safetensors.torch import save_file
            dump_dir.mkdir(parents=True, exist_ok=True)
            save_file({"noise": latents.detach().cpu().contiguous()},
                      dump_dir / "initial_noise.safetensors")

            def tensor_output(output):
                if hasattr(output, "query_embeds"):
                    return output.query_embeds
                if hasattr(output, "hidden_states"):
                    return output.hidden_states
                if isinstance(output, (tuple, list)):
                    return output[0]
                return output

            def capture_text_stage(name):
                def hook(_module, _inputs, output):
                    if name not in captured_text_stages:
                        value = tensor_output(output)
                        captured_text_stages[name] = value.detach().cpu().contiguous()
                return hook

            text_model = pipeline.text_encoder.model.language_model
            text_hooks = [
                text_model.word_embeddings.register_forward_hook(
                    capture_text_stage("token_embeddings")
                ),
                pipeline.queryformer.register_forward_hook(
                    capture_text_stage("queryformer")
                ),
            ]
            for index, layer in enumerate(text_model.layers):
                text_hooks.append(
                    layer.register_forward_hook(capture_text_stage(f"backbone_layer_{index}"))
                )
                if index == 0:
                    # The first dense layer is the earliest place where tiny
                    # BF16 backend differences can cross the layer-1 MoE
                    # routing threshold.  Capture its module boundaries so the
                    # native implementation can be aligned without treating a
                    # later expert-selection mismatch as the root cause.
                    for name, module in (
                        ("layer_0_input_norm", layer.input_layernorm),
                        ("layer_0_qkv", layer.attention.query_key_value),
                        ("layer_0_q_norm", layer.attention.query_layernorm),
                        ("layer_0_k_norm", layer.attention.key_layernorm),
                        ("layer_0_attention_dense", layer.attention.dense),
                        ("layer_0_post_attention_norm", layer.post_attention_layernorm),
                        ("layer_0_mlp_gate", layer.mlp.gate_proj),
                        ("layer_0_mlp_up", layer.mlp.up_proj),
                        ("layer_0_mlp_down", layer.mlp.down_proj),
                    ):
                        text_hooks.append(
                            module.register_forward_hook(capture_text_stage(name))
                        )
                gate = getattr(getattr(layer, "mlp", None), "gate", None)
                if gate is not None:
                    def capture_router(layer_index):
                        def hook(_module, _inputs, output):
                            indices, weights, logits = output
                            captured_text_stages.setdefault(
                                f"router_logits_layer_{layer_index}",
                                logits.detach().to(torch.float32).cpu().contiguous(),
                            )
                            captured_text_stages.setdefault(
                                f"router_indices_layer_{layer_index}",
                                indices.detach().to(torch.float32).cpu().contiguous(),
                            )
                            captured_text_stages.setdefault(
                                f"router_weights_layer_{layer_index}",
                                weights.detach().to(torch.float32).cpu().contiguous(),
                            )
                        return hook

                    text_hooks.append(gate.register_forward_hook(capture_router(index)))
            text_hooks.append(
                text_model.norm.register_forward_hook(
                    capture_text_stage("backbone_final")
                )
            )
            for index, layer in enumerate(pipeline.text_projection.layers):
                text_hooks.append(
                    layer.register_forward_hook(capture_text_stage(f"projection_layer_{index}"))
                )
            text_hooks.append(
                pipeline.text_projection.projector.register_forward_hook(
                    capture_text_stage("projection_final")
                )
            )
            try:
                with torch.inference_mode():
                    conditioning = pipeline.encode_prompt(
                        prompt=args.prompt,
                        negative_prompt=None,
                        do_classifier_free_guidance=args.guidance_scale > 1.0,
                        num_images_per_prompt=1,
                        max_sequence_length=2048,
                        device=pipeline.transformer.device,
                    )
            finally:
                for hook in text_hooks:
                    hook.remove()
            save_file({
                "conditioning": conditioning[0].detach().cpu().contiguous(),
                "attention_mask": conditioning[1].detach().cpu().contiguous(),
            }, dump_dir / "conditioning.safetensors")
            for name, value in captured_text_stages.items():
                save_file({"tensor": value}, dump_dir / f"text_{name}.safetensors")

            def capture_stage(name):
                def hook(_module, _inputs, output):
                    # Only the first denoise step is needed to find the earliest
                    # divergent Transformer component.  Keeping later steps out
                    # avoids a multi-gigabyte diagnostic dump.
                    if name not in captured_transformer_stages:
                        captured_transformer_stages[name] = output.detach().cpu().contiguous()
                return hook

            transformer = pipeline.transformer
            stage_hooks.append(
                transformer.all_x_embedder["1-1"].register_forward_hook(
                    capture_stage("x_embedder")
                )
            )
            for index, layer in enumerate(transformer.noise_refiner):
                stage_hooks.append(
                    layer.register_forward_hook(capture_stage(f"noise_refiner_{index}"))
                )
            stage_hooks.append(
                transformer.cap_embedder.register_forward_hook(capture_stage("cap_embedder"))
            )
            for index, layer in enumerate(transformer.context_refiner):
                stage_hooks.append(
                    layer.register_forward_hook(capture_stage(f"context_refiner_{index}"))
                )
            for index, layer in enumerate(transformer.layers):
                stage_hooks.append(
                    layer.register_forward_hook(capture_stage(f"layer_{index}"))
                )
            stage_hooks.append(
                transformer.all_final_layer["1-1"].register_forward_hook(
                    capture_stage("final_layer")
                )
            )

        def callback(_pipeline, index, _timestep, values):
            if dump_dir:
                from safetensors.torch import save_file
                captured_latents.append(values["latents"].detach().cpu().contiguous())
                save_file({"latent": values["latents"].detach().cpu().contiguous()},
                          dump_dir / f"latent_{index + 1}.safetensors")
            return values

        original_transformer_forward = pipeline.transformer.forward

        def capture_transformer_forward(*forward_args, **forward_kwargs):
            result = original_transformer_forward(*forward_args, **forward_kwargs)
            if dump_dir:
                # Match the exact tensor passed into scheduler.step below.  The
                # transformer returns a list of [C,F,H,W] samples and the
                # pipeline stacks, removes F=1, promotes to FP32, and negates.
                model_output = -torch.stack(result.sample, dim=0).squeeze(2).float()
                captured_model_outputs.append(model_output.detach().cpu().contiguous())
            return result

        kwargs = {
            "prompt": None,
            "prompt_embeds": conditioning[0],
            "prompt_attention_mask": conditioning[1],
            "negative_prompt_embeds": conditioning[2],
            "negative_prompt_attention_mask": conditioning[3],
            "generation_mode": args.generation_mode,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "latents": latents,
            "callback_on_step_end": callback,
        }
        if args.generation_mode == "editing":
            kwargs["image"] = Image.open(args.input_image).convert("RGB")
        start = time.perf_counter()
        pipeline.transformer.forward = capture_transformer_forward
        try:
            with torch.inference_mode():
                image = pipeline(**kwargs).images[0]
        finally:
            pipeline.transformer.forward = original_transformer_forward
            for hook in stage_hooks:
                hook.remove()
        if dump_dir:
            from safetensors.torch import save_file
            for index, model_output in enumerate(captured_model_outputs):
                save_file({"model_output": model_output},
                          dump_dir / f"model_output_{index + 1}.safetensors")
            for name, value in captured_transformer_stages.items():
                save_file({"tensor": value}, dump_dir / f"stage_{name}.safetensors")
            # The released scheduler is stochastic.  Recover and persist the
            # newly sampled noise from its public update equation so a native
            # implementation can distinguish Transformer parity from sampler
            # parity.  The final step has sigma_next=0 and no recoverable noise.
            schedule = pipeline.scheduler.sigmas.detach().cpu().float()
            previous = latents.detach().cpu().float()
            for index, (model_output, next_latent) in enumerate(
                zip(captured_model_outputs, captured_latents)
            ):
                sigma = schedule[index]
                sigma_next = schedule[index + 1]
                if float(sigma_next) > 0.0:
                    x0 = previous - sigma * model_output
                    step_noise = (next_latent.float() - (1.0 - sigma_next) * x0) / sigma_next
                    save_file({"noise": step_noise.contiguous()},
                              dump_dir / f"scheduler_noise_{index + 1}.safetensors")
                previous = next_latent.float()
        # Force a stable filesystem boundary before timing the request.
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return time.perf_counter() - start

    warmup_seconds = []
    for index in range(max(0, args.warmup)):
        warmup_output = args.output.with_name(f".warmup-{index}.png")
        warmup_seconds.append(generate(warmup_output, args.seed + index))
        if not args.keep_warmup:
            warmup_output.unlink(missing_ok=True)
    request_seconds = generate(args.output, args.seed, args.dump_dir)

    report = {
        "schema_version": 1,
        "reference": "inclusionAI/LLaDA-Image-Turbo",
        "model": str(model),
        "device": str(pipeline.transformer.device),
        "dtype": "bfloat16",
        "generation_mode": args.generation_mode,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "initial_noise": "cpu-seeded-fp32-bf16-rounded",
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "request_seconds": request_seconds,
        "output": str(args.output.resolve()),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
