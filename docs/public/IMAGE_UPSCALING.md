# Image upscaling

TurboCider supports **2× and 4×** super-resolution using local Real-ESRGAN x2plus / x4plus Core ML models. Inference, image decoding, tiling and PNG export use Swift and Apple frameworks; the App's upscaling backend requires no Python, PyTorch or coremltools.

## Use

1. Open **模型**, choose the **超分** category, then select **Real-ESRGAN x2plus** or **x4plus**. Use **下载模型** to save the ZIP in your browser, unzip it, and choose **导入并校验本地模型…**. The App checks the actual scale and prepares the Core ML package natively; importing a model never enables automatic upscaling.
2. Go to **创作 → 单图修改 → 图像超分**. Choose the input image and x2/x4 model. There is no separate sidebar item. **管理模型** opens the matching model-center entry; weights, paths and download instructions are managed there.
3. Select **GPU** or **ANE 优先** in the right parameter panel. **自动预加载**, **预加载** and **释放** manage the resident model. Click **开始超分** to create a separate PNG/history record while preserving the original.
4. For image generation, use the selector beside **生成图像**: **不超分**, **超分 ×2**, or **超分 ×4**. New drafts default to no upscaling. Cancellation or failure during upscaling preserves the original generated image. An existing result's **… → 单独超分此图…** action opens the same single-image upscaling workspace with its source filled in and waits for Start.

The two upscalers are auxiliary entries in the unified `library.json` installation index. External packages remain in place; removing a registration retains its files. Exported model configurations include `real-esrgan-x2plus` and `real-esrgan-x4plus` in `modelPaths`; importing them restores upscaler paths without changing the generation engine or opting into post-generation processing. Existing App upscaler paths migrate into the library on startup.

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
