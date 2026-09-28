# Image upscaling

TurboCider supports **2× and 4×** super-resolution using local Real-ESRGAN x2plus / x4plus Core ML models. Inference, image decoding, tiling and PNG export use Swift and Apple frameworks; the App's upscaling backend requires no Python, PyTorch or coremltools.

## Use

1. Open **图像超分** in the sidebar. Its right parameter panel contains model, device and preload settings; the left side contains the image input and result. Select **x2plus** or **x4plus**, follow the model download link, unzip it and select its `.mlpackage`. The App reads the model in place and remembers a separate path for each variant.
2. Select **GPU** or **ANE 优先**. Automatic preload compiles, loads and warms the selected model in advance. The status changes to **已预加载** when ready. One resident model is retained for subsequent images; changing model or device replaces it. **释放超分模型** releases that session.
3. Choose an image and click **开始超分**. The original is preserved; the PNG result receives a separate history record.
4. In **创作**, enable **生成后自动超分** in the right parameter panel to upscale each successful image generation. The Generate menu and result toolbar also offer upscaling. Cancellation or failure during upscaling preserves the original generated image.

Preloading moves startup work earlier; it does not eliminate startup time or persist a loaded model across App restarts. It temporarily reserves the App's job slot so generation and preload cannot compete. A Core ML prediction in progress finishes before cancellation takes effect.

## Models and download links

| Model | Output size | Preconverted FP16 package |
| --- | --- | --- |
| x2plus | 2× each dimension | [Download x2plus](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x2plus_522_fp16.zip) |
| x4plus | 4× each dimension | [Download x4plus](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x4plus_522_fp16.zip) |

Both archives are about 30 MiB. App links also open the [model conversion and processing reference](https://github.com/hanxiao/real-esrgan-coreml#usage). These packages are already converted: using them in TurboCider needs no Python conversion step. Building new exports from original PyTorch checkpoints remains an external developer workflow. Weights originate from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3-Clause and remain outside the App bundle and Git repository. No model downloads start automatically.

## Runtime contract

- One RGB NCHW tensor input `[1,3,N,N]` and one 2× or 4× RGB output; FP16 or FP32 interfaces, fixed square input size between 64 and 1024. The tested packages use N=522 and FP16 input/output. Scale is validated from the loaded model, rather than assumed from its filename.
- Default **GPU** uses Core ML CPU/GPU. **ANE 优先** allows CPU/ANE and excludes GPU; unsupported operators may use CPU. This selection is independent of generation's ANE toggle and does not prove that every operator runs on ANE.
- Preload includes one zero-input prediction; actual image requests reuse the same model. Cache identity includes resolved path, file sizes/modification times and device choice. Only one model is retained, and explicit release also works from the global memory control.
- Overlapping tiles with reflected padding and feathered joins support rectangular images. ImageIO normalizes orientation; alpha is resized separately and preserved.
- One static image per job, at most 64 Mi output pixels. Video and animated inputs are rejected. Output is staged, decoded and checked before atomic publication; the original is never overwritten.

GPU has lower startup overhead on the tested M4 Pro. With the model preloaded, CPU/ANE can complete image processing faster. See [local measurements and verification](../status/dev-verify-upscaling-2026-09-28.md); timings depend on model, input dimensions and machine. x2 and x4 deliver different output resolutions, so their speed comparison is not an equal-resolution quality comparison.

Z-Image Turbo defaults to **8 sampling steps**. Saved drafts retain explicit settings. The separately selected local experimental streaming preset remains its measured 9-step preset.
