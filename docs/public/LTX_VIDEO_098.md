# LTX-Video 0.9.8 image-to-video (experimental)

This route runs `ltxv-2b-0.9.8-distilled.safetensors` natively on the GPU with MLX / Metal; it does not require a ComfyUI or Python inference service. The checkpoint contains the transformer and video VAE and requires a separate T5-XXL encoder and tokenizer. The multiscale path also requires the official 0.9.8 spatial latent upscaler. T5 uses BF16 weights and FP32 intermediate computation; the transformer and VAE use BF16 and the scheduler uses FP32. Model weights have their own license: review the publisher's terms, including commercial-use restrictions, before use.

## Model directory

Install **LTX-Video 0.9.8 2B Distilled** from the App's model library using the Hugging Face source, or register an existing directory containing:

```text
model/
├── ltxv-2b-0.9.8-distilled.safetensors
├── ltxv-spatial-upscaler-0.9.8.safetensors  # for multiscale
├── text_encoders/
│   └── t5xxl_fp16.safetensors
└── tokenizer/
    └── tokenizer.json
```

The library recipe obtains the checkpoint and spatial upscaler from Lightricks/LTX-Video, T5-XXL from comfyanonymous/flux_text_encoders, and the tokenizer from google-t5/t5-base. The upscaler is about 505 MB; the eight-step single-scale path still works without it. See `examples/requests/ltx098-model-library.json` for the download request.

## Generate

In the App, select this model and image-to-video, import one first-frame image, enter a prompt, and select resident GPU execution. The default multiscale workflow runs seven denoising steps, learned 2× latent upsampling and per-channel AdaIN statistics matching, then three refinement steps. At 24 fps, 121 frames produce approximately 5.04 seconds of silent video. The input is center-cropped to the base aspect ratio before resizing, and the output path must not already exist.

For multiscale, width and height specify a **base canvas**, not necessarily the final file dimensions. The official 0.6666666 downscale and 32-pixel alignment turn a 512×512 base into 320×320 for the first pass and 640×640 for the second pass. The App retains the 640×640 output by default; enabling the final resize produces 512×512, as in the official example. This final bilinear resize does not change the generated motion. The previous eight-step single-scale route remains available and produces the base dimensions without the upscaler. Strength 1 locks the first-frame condition; later frames are still generated.

For the CLI, replace the image and output paths in `examples/requests/ltx098-image-to-video.json` with absolute paths, then run:

```sh
dist/cli/turbocider generate /absolute/path/to/model examples/requests/ltx098-image-to-video.json
```

The request uses model ID `ltx-video-0.9.8-distilled`, operation `video.image`, and one `first_frame` image input with strength in 0–1. `steps: 10` selects multiscale; set `ltx098_resize_to_base: true` for the official final resize (default false). `steps: 8` selects the previous single-scale path. Base dimensions are multiples of 32 from 64 to 1024; frame count is `8n+1` from 9 to 121 at 24 fps. The base is limited to 512×512×121 pixel-frames, and the upscaled stage to 6400 latent tokens and 640×640×121 pixel-frames. Result `width` and `height` are the actual file size; `requested_base_width` and `requested_base_height` record the input canvas.

This is experimental: audio, LoRA, streamed loading, automatic prompt enhancement and ANE execution are not supported, and LTX 2.5-specific acceleration options do not apply. Performance and visual results vary by hardware, image, prompt, seed and video encoder. Multiscale does not guarantee better results than eight-step single-scale, nor perfect face, limb or motion consistency; backends and decoder noise are not bit-identical.

Building into a separate `build/` directory does not update an existing `dist/TurboCider.app`; use the project's packaging workflow to build the version you intend to run.

See the [LTX 0.9.8 benchmark record](../benchmark/LTX_VIDEO_098.md) for the measurement boundaries, failed quality gates and ANE component experiments.
